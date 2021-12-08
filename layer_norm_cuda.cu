#include<iostream>
#include<cub/block/block_reduce.cuh>

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

const int kBlockSize = 128;
const int kThreadElements = 4;

int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template<typename T>
void PrepareAlloc(T **x, int size, bool use_host, bool human_readable,
                  int seed=99, int init=-1) {
  srand(seed);
  int max_int = 32768;
  T *buf_x = new T[size];
  for (int i = 0; i < size; i++) {
    if (init != -1) {
      buf_x[i] = init;
    } else if (human_readable) {
      buf_x[i] = i / 10.;
    } else {
      buf_x[i] = static_cast<T>(static_cast<float>(rand() % max_int) / max_int);
    }
  }
  if (use_host) {
    checkCUDA(cudaMallocManaged(&(*x), size * sizeof(T)));
    for (int i = 0; i < size; i++) {
      (*x)[i] = buf_x[i];
    }
  } else {
    checkCUDA(cudaMalloc(&(*x), size * sizeof(T)));
    checkCUDA(cudaMemcpy(*x, buf_x,  size * sizeof(T), cudaMemcpyHostToDevice));
  }

  delete[] buf_x;
}

template<typename T>
void Print2D(const T* x, int N, int D, std::string msg) {
  printf("%s\n", msg.c_str());
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      printf("%f, ", static_cast<float>(x[j + i * D]));
    }
    printf("\n");
  }
}

template<typename T>
void IsClose2D(const T* x, const T* y, int N, int D, std::string msg) {
  bool is_same = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      float d_val = static_cast<float>(x[j + i * D]);
      float h_val = static_cast<float>(y[j + i * D]);
      if (abs(d_val - h_val > 0.001f)) {
        is_same = false;
        printf("Found diff: CPU=%f, GPU=%f at (%d, %d)\n", h_val, d_val, i, j);
        break;
      }
    }
    if (!is_same) break;
  }
  printf("Test (%s): %s\n", msg.c_str(), is_same ? "True" : "False");
}

template<typename T, typename U>
__host__ __device__ U GetAs(const T* __restrict__ in, int offset) {
  return static_cast<U>(in[offset]);
}

template<typename T, typename U>
__device__ void GetStatsNaive(const T* __restrict__ row, const U epsilon,
                              const int tid, const int D, U &mean, U &ivar,
                              int t) {
  U sum = 0;
  for (int i = 0; i < D; i++) {
    sum += GetAs<T, U>(row, i);
  }
  mean = sum / D;
  U sum_ivar = 0;
  for (int i = 0; i < D; i++) {
    U curr = GetAs<T, U>(row, i);
    sum_ivar += (curr - mean) * (curr - mean);
  }
  ivar = rsqrt(sum_ivar / D + epsilon);
}

// Essentially, this is a row reduction, which will be conducted twice for the
// mean and ivar respectively.
template<typename T, typename U>
__device__ void GetStats(const T* __restrict__ row, const U epsilon,
                         const int tid, const int D, U &mean, U &ivar) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
      typename BlockReduce::TempStorage reduce;
      U broadcast[1];
  } temp_storage;

  U thread_data[kThreadElements];

  const int workload_per_round = kBlockSize * kThreadElements;
  const int rounds = (D + workload_per_round - 1) / workload_per_round;

  int i = tid;
  U sum = 0;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      if (i * kThreadElements + j < D) {
        thread_data[j] = GetAs<T, U>(row, i * kThreadElements + j);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
    // It seems we have to add a sync here to ensure the data are loaded into
    // thread_data.
    __syncthreads();
    U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
    sum += aggregate;
    i += kBlockSize;
  }
  
  if (tid == 0) {
    temp_storage.broadcast[0] = sum;
  }
  __syncthreads();
  mean = temp_storage.broadcast[0] / D;

  i = tid;
  U sum_ivar = 0;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      if (i * kThreadElements + j < D) {
        U curr = GetAs<T, U>(row, i * kThreadElements + j);
        thread_data[j] = (curr - mean) * (curr - mean);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
    // It seems we have to add a sync here to ensure the data are loaded into
    // thread_data.
    __syncthreads();
    U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
    sum_ivar += aggregate;
    i += kBlockSize;
  }
  
  if (tid == 0) {
    temp_storage.broadcast[0] = sum_ivar;
  }
  __syncthreads();

  ivar = rsqrt(temp_storage.broadcast[0] / D + epsilon);
}

// Part1: compute the mean and ivar and store them to a cache.
template<typename T, typename U>
__global__ void LayerNormKernelPart1(const T* __restrict__ x, const U epsilon,
                                     const int N, const int D,
                                     U* __restrict__ cache_ivar,
                                     U* __restrict__ cache_mean) {
  // Assume gridDim.x == N.
  U mean, ivar;
  GetStats(x + blockIdx.x * D, epsilon, threadIdx.x, D, mean, ivar);
  // Intermediate results to speedup backprop.
  if (threadIdx.x == 0) {
    cache_ivar[blockIdx.x] = ivar;
    cache_mean[blockIdx.x] = mean;
  }
}

// Part2: compute the normalized values.
template<typename T, typename U>
__global__ void LayerNormKernelPart2(const T* __restrict__ x,
                                     const U* __restrict__ gamma,
                                     const U* __restrict__ beta,
                                     const U* __restrict__ cache_ivar,
                                     const U* __restrict__ cache_mean,
                                     const U epsilon,
                                     const int N, const int D,
                                     T* __restrict__ y) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * D) return;

  const int col = tid % D;
  const int row = tid / D;

  U mean = cache_mean[row];
  U ivar = cache_ivar[row];
  U curr = GetAs<T, U>(x, tid);
  y[tid] = static_cast<T>((curr - mean) * ivar * gamma[col] + beta[col]);
}

template<typename T, typename U>
void LayerNormGPU(const T* x, const U* gamma, const U* beta, const U epsilon,
                  const int N, const int D, T* y, U* cache_ivar,
                  U* cache_mean) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  LayerNormKernelPart1<<<N, kBlockSize>>>(x, 0.001f, N, D, cache_ivar,
                                          cache_mean);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (y) p1: %f ms\n", milliseconds);

  cudaEventRecord(start);
  LayerNormKernelPart2<<<div_up(N * D, kBlockSize), kBlockSize>>>(
      x, gamma, beta, cache_ivar, cache_mean, 0.001f, N, D, y);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (y) p2: %f ms\n", milliseconds);
}

template<typename T, typename U>
void LayerNormCPU(const T* x, const U* gamma, const U* beta, const U epsilon,
                  const int N, const int D, T* y) {
  for(int j = 0; j < N; j++) {
    // Part1.
    U mean, ivar;
    U sum = 0;
    for(int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum += curr;
    }
    mean = sum / D;
    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = rsqrt(sum_ivar / D + epsilon);
    // printf("%f\n", mean);
    // printf("%f\n", ivar);

    // Part2.
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      y[j * D + i] = static_cast<T>((curr - mean) * ivar * gamma[i] + beta[i]);
    }
  }
}

// To be replaced with "SetZero<T>".
template<typename U>
__global__ void InitGradBetaGamma(U* __restrict__ dgamma, U* __restrict__ dbeta,
                                  int D) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= D) return;

  dgamma[tid] = 0;
  dbeta[tid] = 0;
}

// To be replaced with "LaunchColumnReduction".
// TODO(kaixih): How to prepare the custom op.
template<typename T, typename U>
__global__ void LayerNormGradBetaGammaAtomic(const T* __restrict__ dy,
                                             const T* __restrict__ x,
                                             const U* __restrict__ cache_mean,
                                             const U* __restrict__ cache_ivar,
                                             const int N, const int D,
                                             U* __restrict__ dgamma,
                                             U* __restrict__ dbeta) {
  const int tid = threadIdx.x;
  const int row_stride = blockDim.x;
  const int col_stride = gridDim.x;

  for (int j = blockIdx.x; j < N; j += col_stride) {
    for (int i = tid; i < D; i += row_stride) {
      U dy_curr = GetAs<T, U>(dy, j * D + i);
      atomicAdd(dgamma + i,
                dy_curr * (x[j * D + i] - cache_mean[j]) * cache_ivar[j]);
      atomicAdd(dbeta + i, dy_curr);
    }
  }
}

template<typename T, typename U>
__global__ void LayerNormGradBetaGamma(const T* __restrict__ dy,
                                       const T* __restrict__ x,
                                       const U* __restrict__ cache_mean,
                                       const U* __restrict__ cache_ivar,
                                       const int N, const int D,
                                       U* __restrict__ dgamma,
                                       U* __restrict__ dbeta) {
  // Assume the total thread number == D.
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= D) return;

  U sum_dgamma = 0;
  U sum_dbeta = 0;
  for (int i = 0; i < N; i++) {
    U dy_curr = GetAs<T, U>(dy, i * D + tid);
    sum_dgamma += dy_curr * (x[i * D + tid] - cache_mean[i]) * cache_ivar[i];
    sum_dbeta += dy_curr;
  }

  dgamma[tid] = sum_dgamma;
  dbeta[tid] = sum_dbeta;
}

// Part1: compute the dl_dvars and dl_dmu and store them to a cache.
template<typename T, typename U>
__global__ void LayerNormGradInputPart1(const T* __restrict__ dy,
                                        const T* __restrict__ x,
                                        const U* __restrict__ gamma,
                                        const U* __restrict__ cache_mean,
                                        const U* __restrict__ cache_ivar,
                                        const int N, const int D,
                                        U* dl_dvars, U *dl_dmus) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
      typename BlockReduce::TempStorage reduce;
      U broadcast[1];
  } temp_storage;
  U thread_data[kThreadElements];

  const int workload_size = kBlockSize * kThreadElements;
  const int rounds = (D + workload_size - 1) / workload_size;

  int i = tid;
  int k = blockIdx.x;
  U dl_dvar = 0;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      int row_offset = i * kThreadElements + j;
      if (row_offset < D) {
        U curr = GetAs<T, U>(dy, k * D + row_offset);
        thread_data[j] = curr * gamma[row_offset] *
                         (x[k * D + row_offset] - cache_mean[k]) *
                         (-0.5) * (cache_ivar[k] * cache_ivar[k] *
                                   cache_ivar[k]);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
    __syncthreads();
    U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
    dl_dvar += aggregate;
    i += kBlockSize;
  }
  
  if (tid == 0) {
    temp_storage.broadcast[0] = dl_dvar;
  }
  __syncthreads();
  dl_dvar = temp_storage.broadcast[0];
  dl_dvars[blockIdx.x] = dl_dvar;

  i = tid;
  U dl_dmu = 0;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      int row_offset = i * kThreadElements + j;
      if (row_offset < D) {
        U curr = GetAs<T, U>(dy, k * D + row_offset);
        thread_data[j] = -1. * curr * gamma[row_offset] *
                         cache_ivar[k] + dl_dvar * (-2. / D) *
                         (x[k * D + row_offset] - cache_mean[k]);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
    __syncthreads();
    U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
    dl_dmu += aggregate;
    i += kBlockSize;
  }
  
  if (tid == 0) {
    temp_storage.broadcast[0] = dl_dmu;
  }
  __syncthreads();
  dl_dmu = temp_storage.broadcast[0];
  dl_dmus[k] = dl_dmu;
}

// Part2: compute the normalized values.
template<typename T, typename U>
__global__ void LayerNormGradInputPart2(const T* __restrict__ dy,
                                        const T* __restrict__ x,
                                        const U* __restrict__ gamma,
                                        const U* __restrict__ cache_mean,
                                        const U* __restrict__ cache_ivar,
                                        const U* dl_dvars, const U *dl_dmus,
                                        const int N, const int D,
                                        T * dx) {

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * D) return;

  const int col = tid % D;
  const int row = tid / D;

  U curr = GetAs<T, U>(dy, row * D + col);
  U dl_di = curr * gamma[col] * cache_ivar[row];
  U di_dx = 1.;
  U dvar_dx = 2. * (x[row * D + col] - cache_mean[row]) / D;
  U dmu_dx = 1. / D;
  U dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
  dx[row * D + col] = static_cast<T>(dl_dx);
}

template<typename T, typename U>
void LayerNormGradGPU(const T* dy, const T* x, const U* cache_mean,
                      const U* cache_ivar, const U* gamma, const int N,
                      const int D, U* temp_1, U* temp_2, T* dx, U* dgamma,
                      U* dbeta) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  InitGradBetaGamma<<<div_up(D, kBlockSize), kBlockSize>>>(dgamma, dbeta, D);

  if (N <= D) {
    LayerNormGradBetaGamma<<<div_up(D, kBlockSize), kBlockSize>>>(
        dy, x, cache_mean, cache_ivar, N, D, dgamma, dbeta);
  } else {
    LayerNormGradBetaGammaAtomic<<<N, kBlockSize>>>(
        dy, x, cache_mean, cache_ivar, N, D, dgamma, dbeta);
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dgamma, dbeta): %f ms\n", milliseconds);

  cudaEventRecord(start);
  LayerNormGradInputPart1<<<N, kBlockSize>>>(
      dy, x, gamma, cache_mean, cache_ivar, N, D, temp_1, temp_2);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dx) p1: %f ms\n", milliseconds);

  cudaEventRecord(start);
  LayerNormGradInputPart2<<<div_up(N * D, kBlockSize), kBlockSize>>>(
      dy, x, gamma, cache_mean, cache_ivar, temp_1, temp_2, N, D, dx);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dx) p2: %f ms\n", milliseconds);
}

template<typename T, typename U>
void LayerNormGradCPU(const T* dy, const T* x, const U* cache_mean,
                      const U* cache_ivar, const U* gamma, const int N,
                      const int D, T* dx_h, U* dgamma_h, U* dbeta_h) {
  // Compute dgamma, dbeta.
  for (int i = 0; i < D; i++) {
    dgamma_h[i] = 0;
    dbeta_h[i] = 0;
    for (int j = 0 ; j < N; j++) {
      U dy_curr = static_cast<U>(dy[j * D + i]);
      dgamma_h[i] += dy_curr * (x[j * D + i] - cache_mean[j]) * cache_ivar[j];
      dbeta_h[i] += dy_curr;
    }
  }

  for (int i = 0; i < N; i++) {
    // Part1.
    U dl_dvar = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dvar += curr * gamma[j] * (x[i * D + j] - cache_mean[i]) * (-0.5) *
                     (cache_ivar[i] * cache_ivar[i] * cache_ivar[i]);
    }
    U dl_dmu = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dmu += -1. * curr * gamma[j] * cache_ivar[i];
      dl_dmu += dl_dvar * (-2. / D) * (x[i * D + j] - cache_mean[i]);
    }

    // Part2.
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      U dl_di = curr * gamma[j] * cache_ivar[i];
      U di_dx = 1.;
      U dvar_dx = 2. * (x[i * D + j] - cache_mean[i]) / D;
      U dmu_dx = 1. / D;
      U dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmu * dmu_dx;
      dx_h[i * D + j] = static_cast<T>(dx);
    }
  }
}



#define DTYPE float

int main() {

  /** Parameters and Knobs **/
  int N = 10000000;
  int D = 10;
  bool allow_print = false;
  bool human_readable = false;
  bool use_host = false;

  DTYPE* x;
  float* gamma;
  float* beta;
  DTYPE* y;
  float* cache_ivar;
  float* cache_mean;

  PrepareAlloc(&x, N * D, use_host, human_readable, 12);
  PrepareAlloc(&gamma, D, use_host, human_readable, 13);
  PrepareAlloc(&beta, D, use_host, human_readable, 14);
  PrepareAlloc(&y, N * D, use_host, human_readable);

  PrepareAlloc(&cache_ivar, N, use_host, human_readable);
  PrepareAlloc(&cache_mean, N, use_host, human_readable);

  LayerNormGPU(x, gamma, beta, 0.001f, N, D, y, cache_ivar, cache_mean);
  checkCUDA(cudaDeviceSynchronize());
  if (use_host && allow_print) {
    Print2D(y, N, D, "GPU y:");
  }

  if (use_host) {
    DTYPE *y_h = new DTYPE[N * D];
    LayerNormCPU(x, gamma, beta, 0.001f, N, D, y_h);
    if (allow_print) {
      Print2D(y_h, N, D, "CPU y:");
    }

    IsClose2D(y, y_h, N, D, "y");
    delete[] y_h;
  }
  // ---- Forward Done Here ----

  DTYPE* dy;
  DTYPE* dx;
  float* dgamma;
  float* dbeta;

  float* temp1;
  float* temp2;

  PrepareAlloc(&dy, N * D, use_host, human_readable, 99, 1);
  PrepareAlloc(&dx, N * D, use_host, human_readable);
  PrepareAlloc(&dgamma, D, use_host, human_readable);
  PrepareAlloc(&dbeta, D, use_host, human_readable);

  PrepareAlloc(&temp1, N, use_host, human_readable);
  PrepareAlloc(&temp2, N, use_host, human_readable);

  LayerNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, D, temp1, temp2, dx,
                   dgamma, dbeta);
  checkCUDA(cudaDeviceSynchronize());
  if (use_host && allow_print) {
    Print2D(dgamma, 1, D, "GPU dgamma:");
    Print2D(dbeta, 1, D, "GPU dbeta:");
    Print2D(dx, N, D, "GPU dx:");
  }

  if (use_host) {
    DTYPE *dx_h = new DTYPE[N * D];
    float *dgamma_h = new float[D];
    float *dbeta_h = new float[D];
    LayerNormGradCPU(
        dy, x, cache_mean, cache_ivar, gamma, N, D, dx_h, dgamma_h, dbeta_h);
    if (allow_print) {
      Print2D(dgamma_h, 1, D, "CPU dgamma:");
      Print2D(dbeta_h, 1, D, "CPU dbeta:");
      Print2D(dx_h, N, D, "CPU dx:");
    }

    IsClose2D(dgamma, dgamma_h, 1, D, "dgamma");
    IsClose2D(dbeta, dbeta_h, 1, D, "dbeta");
    IsClose2D(dx, dx_h, N, D, "dx");

    delete[] dx_h;
    delete[] dgamma_h;
    delete[] dbeta_h;
  }

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(gamma));
  checkCUDA(cudaFree(beta));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(dy));
  checkCUDA(cudaFree(dx));
  checkCUDA(cudaFree(dgamma));
  checkCUDA(cudaFree(dbeta));
  checkCUDA(cudaFree(cache_mean));
  checkCUDA(cudaFree(cache_ivar));
  checkCUDA(cudaFree(temp1));
  checkCUDA(cudaFree(temp2));
}
