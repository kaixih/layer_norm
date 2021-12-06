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

const int kBlockSize = 256;
const int kThreadElements = 4;

template<typename T, typename U>
__host__ __device__ U GetData(const T* __restrict__ in, int offset) {
  return static_cast<U>(in[offset]);
}

template<typename T, typename U>
__device__ void GetStats(const T* __restrict__ row, const U epsilon,
                         U &mean, U &ivar, int tid, int D) {
  U sum = 0;
  for (int i = 0; i < D; i++) {
    sum += GetData<T, U>(row, i);
  }
  mean = sum / D;
  U sum_ivar = 0;
  for (int i = 0; i < D; i++) {
    U curr = GetData<T, U>(row, i);
    sum_ivar += (curr - mean) * (curr - mean);
  }
  ivar = rsqrt(sum_ivar / D + epsilon);
}

template<typename T, typename U>
__device__ void GetStatsV2(const T* __restrict__ row, const U epsilon,
                           U &mean, U &ivar, int tid, int D) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
      typename BlockReduce::TempStorage reduce;
      U broadcast[1];
  } temp_storage;
  U thread_data[kThreadElements];

  U sum = 0;
  const int workload_size = kBlockSize * kThreadElements;
  const int rounds = (D + workload_size - 1) / workload_size;
  int i = tid;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      if (i * kThreadElements + j < D) {
        thread_data[j] = GetData<T, U>(row, i * kThreadElements + j);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
    U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
    sum += aggregate;
    i += kBlockSize;
  }
  
  if (tid == 0) {
    temp_storage.broadcast[0] = sum;
  }
  __syncthreads();
  mean = temp_storage.broadcast[0] / D;

  U sum_ivar = 0;

  i = tid;
  for (int round = 0; round < rounds; round++) {
    for (int j = 0; j < kThreadElements; j++) {
      if (i * kThreadElements + j < D) {
        U curr = GetData<T, U>(row, i * kThreadElements + j);
        thread_data[j] = (curr - mean) * (curr - mean);
      } else {
        thread_data[j] = static_cast<U>(0);
      }
    }
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

// Like FusedBatchNormV3, we support T: {half, float} and U: {float}.
template<typename T, typename U>
__global__ void LayerNormKernel(const T* __restrict__ x,
                                const U* __restrict__ gamma,
                                const U* __restrict__ beta,
                                const U epsilon,
                                T* __restrict__ y,
                                U* __restrict__ cache_xivar,
                                U* __restrict__ cache_xmu,
                                int N, int D) {
  const int tid = threadIdx.x;
  const int row_stride = blockDim.x;
  const int col_stride = gridDim.x;

  for (int j = blockIdx.x; j < N; j += col_stride) {
    U mean, ivar;
    // GetStats(x + j * D, epsilon, mean, ivar, tid, D);
    GetStatsV2(x + j * D, epsilon, mean, ivar, tid, D);
    for (int i = tid; i < D; i += row_stride) {
      U curr = GetData<T, U>(x, j * D + i);
      y[j * D + i] =
          static_cast<T>((curr - mean) * ivar * gamma[i] + beta[i]);
      // Intermediate results to speedup backprop.
      cache_xmu[j * D + i] = curr - mean;
    }
    cache_xivar[j] = 1.0 / ivar;
  }
}

template<typename T, typename U>
void LayerNormCPU(const T* x, const U* gamma, const U* beta, const U epsilon,
                  T* y, int N, int D) {
  for(int j = 0; j < N; j++) {
    U mean, ivar;
    U sum = 0;
    for(int i = 0; i < D; i++) {
      U curr = GetData<T, U>(x, j * D + i);
      sum += curr;
    }
    mean = sum / D;
    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetData<T, U>(x, j * D + i);
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = rsqrt(sum_ivar / D + epsilon);

    for (int i = 0; i < D; i++) {
      U curr = GetData<T, U>(x, j * D + i);
      y[j * D + i] =
          static_cast<T>((curr - mean) * ivar * gamma[i] + beta[i]);
    }
  }
}

template<typename T, typename U>
void LayerNormGradCPU(T* dy, U* cache_xmu, U* cache_xivar, U* gamma, T* dx_h,
                      U* dgamma_h, U* dbeta_h, int N, int D) {
  // Compute dgamma, dbeta.
  for (int i = 0; i < D; i++) {
    dgamma_h[i] = 0;
    dbeta_h[i] = 0;
    for (int j = 0 ; j < N; j++) {
      U dy_curr = static_cast<U>(dy[j * D + i]);
      dgamma_h[i] += dy_curr * cache_xmu[j * D + i] / cache_xivar[j];
      dbeta_h[i] += dy_curr;
    }
  }

  // Compute dx.
  for (int i = 0; i < N; i++) {
    U dl_dvar = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dvar += curr * gamma[j] * cache_xmu[i * D + j] * (-0.5) *
                     pow(cache_xivar[i], -3);
    }
    U dl_dmu = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dmu += -1. * curr * gamma[j] / cache_xivar[i];
      dl_dmu += dl_dvar * (-2. / D) * cache_xmu[i * D + j];
    }

    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      U dl_di = curr * gamma[j] / cache_xivar[i];
      U di_dx = 1.;
      U dvar_dx = 2. * cache_xmu[i * D + j] / D;
      U dmu_dx = 1. / D;
      U dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmu * dmu_dx;
      dx_h[i * D + j] = static_cast<T>(dx);
    }
  }

}

template<typename T, typename U>
__global__ void LayerNormGradBetaGamma(const T* __restrict__ dy,
                                       const U* __restrict__ cache_xmu,
                                       const U* __restrict__ cache_xivar,
                                       U* __restrict__ dgamma,
                                       U* __restrict__ dbeta, int N, int D) {
  const int tid = threadIdx.x;
  const int row_stride = blockDim.x;
  const int col_stride = gridDim.x;

  for (int j = blockIdx.x; j < N; j += col_stride) {
    for (int i = tid; i < D; i += row_stride) {
      U dy_curr = GetData<T, U>(dy, j * D + i);
      atomicAdd(dgamma + i, dy_curr * cache_xmu[j * D + i] / cache_xivar[j]);
      atomicAdd(dbeta + i, dy_curr);
    }
  }
}

template<typename U>
__global__ void InitGradBetaGamma(U* __restrict__ dgamma, U* __restrict__ dbeta,
                                  int D) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < D) {
    dgamma[tid] = 0;
    dbeta[tid] = 0;
  }
}

template<typename T, typename U>
__global__ void LayerNormGradInput(const T* __restrict__ dy,
                                   const U* __restrict__ gamma,
                                   const U* __restrict__ cache_xmu,
                                   const U* __restrict__ cache_xivar,
                                   T * dx, int N, int D) {
  const int tid = threadIdx.x;
  const int row_stride = blockDim.x;
  const int col_stride = gridDim.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
      typename BlockReduce::TempStorage reduce;
      U broadcast[1];
  } temp_storage;
  U thread_data[kThreadElements];

  const int workload_size = kBlockSize * kThreadElements;
  const int rounds = (D + workload_size - 1) / workload_size;
  for (int k = blockIdx.x; k < N; k += col_stride) {
    int i = tid;
    U dl_dvar = 0;
    for (int round = 0; round < rounds; round++) {
      for (int j = 0; j < kThreadElements; j++) {
        int row_offset = i * kThreadElements + j;
        if (row_offset < D) {
          U curr = GetData<T, U>(dy, k * D + row_offset);
          thread_data[j] = curr * gamma[row_offset] *
                           cache_xmu[k * D + row_offset] *
                           (-0.5) * pow(cache_xivar[k], -3);
        } else {
          thread_data[j] = static_cast<U>(0);
        }
      }
      U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
      dl_dvar += aggregate;
      i += kBlockSize;
    }
    
    if (tid == 0) {
      temp_storage.broadcast[0] = dl_dvar;
    }
    __syncthreads();
    dl_dvar = temp_storage.broadcast[0];

    i = tid;
    U dl_dmu = 0;
    for (int round = 0; round < rounds; round++) {
      for (int j = 0; j < kThreadElements; j++) {
        int row_offset = i * kThreadElements + j;
        if (row_offset < D) {
          U curr = GetData<T, U>(dy, k * D + row_offset);
          thread_data[j] = -1. * curr * gamma[row_offset] /
                           cache_xivar[k] + dl_dvar * (-2. / D) *
                           cache_xmu[k * D + row_offset];
        } else {
          thread_data[j] = static_cast<U>(0);
        }
      }
      U aggregate = BlockReduce(temp_storage.reduce).Sum(thread_data);
      dl_dmu += aggregate;
      i += kBlockSize;
    }
    
    if (tid == 0) {
      temp_storage.broadcast[0] = dl_dmu;
    }
    __syncthreads();
    dl_dmu = temp_storage.broadcast[0];

    for (int i = tid; i < D; i += row_stride) {
      U curr = GetData<T, U>(dy, k * D + i);
      U dl_di = curr * gamma[i] / cache_xivar[k];
      U di_dx = 1.;
      U dvar_dx = 2. * cache_xmu[k * D + i] / D;
      U dmu_dx = 1. / D;
      U dl_dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmu * dmu_dx;
      dx[k * D + i] = static_cast<T>(dl_dx);
    }
  }
}

template<typename T, typename U>
void LayerNormGradGPU(T* dy, U* cache_xmu, U* cache_xivar, U* gamma, T* dx,
                      U* dgamma, U* dbeta, int N, int D) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  dim3 threads_init(128, 1, 1);
  dim3 blocks_init((D + 127) / 128, 1, 1);
  InitGradBetaGamma<<<blocks_init, threads_init>>>(dgamma, dbeta, D);
  dim3 threads(128, 1, 1);
  dim3 blocks(100, 1, 1);
  LayerNormGradBetaGamma<<<blocks, threads>>>(
      dy, cache_xmu, cache_xivar, dgamma, dbeta, N, D);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dgamma, dbeta): %f ms\n", milliseconds);

  cudaEventRecord(start);
  dim3 threads_input(kBlockSize, 1, 1);
  dim3 blocks_input(100, 1, 1);
  LayerNormGradInput<<<blocks_input, threads_input>>>(
      dy, gamma, cache_xmu, cache_xivar, dx, N, D);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dx): %f ms\n", milliseconds);

}

#define DTYPE float

template<typename T>
void PrepareAlloc(T **x, int size, bool use_host, bool human_readable,
                  int init=-1) {
  T *buf_x = new T[size];
  for (int i = 0; i < size; i++) {
    if (init != -1) {
      buf_x[i] = init;
    } else if (human_readable) {
      buf_x[i] = i / 10.;
    } else {
      buf_x[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
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

int main() {

  /** Parameters and Knobs **/
  int N = 1000;
  int D = 10000;
  bool allow_print = false;
  bool human_readable = false;
  bool use_host = true;

  DTYPE* x;
  float* gamma;
  float* beta;
  DTYPE* y;
  float* cache_xivar;
  float* cache_xmu;

  PrepareAlloc(&x, N * D, use_host, human_readable);
  PrepareAlloc(&gamma, D, use_host, human_readable);
  PrepareAlloc(&beta, D, use_host, human_readable);
  PrepareAlloc(&y, N * D, use_host, human_readable);

  PrepareAlloc(&cache_xivar, N, use_host, human_readable);
  PrepareAlloc(&cache_xmu, N * D, use_host, human_readable);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  const dim3 threads(kBlockSize, 1);
  const dim3 blocks(50, 1, 1);
  LayerNormKernel<<<blocks, threads>>>(x, gamma, beta, 0.001f, y, cache_xivar,
                                       cache_xmu, N, D);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time: %f ms\n", milliseconds);

  checkCUDA(cudaDeviceSynchronize());
  if (use_host && allow_print) {
    printf("GPU:\n");
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < D; j++) {
        printf("%f, ", static_cast<float>(y[j + i * D]));
      }
      printf("\n");
    }
  }

  if (use_host) {
    DTYPE *y_h = new DTYPE[N * D];
    LayerNormCPU(x, gamma, beta, 0.001f, y_h, N, D);
    if (allow_print) {
      printf("CPU:\n");
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
          printf("%f, ", static_cast<float>(y_h[j + i * D]));
        }
        printf("\n");
      }
    }

    bool is_same = true;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < D; j++) {
        float d_val = static_cast<float>(y[j + i * D]);
        float h_val = static_cast<float>(y_h[j + i * D]);
        if (abs(d_val - h_val > 0.03f)) {
          is_same = false;
          printf("Found diff: CPU=%f, GPU=%f at (%d, %d)\n", h_val, d_val, i, j);
          break;
        }
      }
    }
    printf("Test pass: %s\n", is_same ? "True" : "False");
    delete[] y_h;
  }

  DTYPE* dy;
  DTYPE* dx;
  float* dgamma;
  float* dbeta;

  PrepareAlloc(&dy, N * D, use_host, human_readable, 1);
  PrepareAlloc(&dx, N * D, use_host, human_readable);
  PrepareAlloc(&dgamma, D, use_host, human_readable);
  PrepareAlloc(&dbeta, D, use_host, human_readable);

  LayerNormGradGPU(dy, cache_xmu, cache_xivar, gamma, dx, dgamma, dbeta, N, D);
  checkCUDA(cudaDeviceSynchronize());
  if (use_host && allow_print) {
    printf("GPU dgamma:\n");
    for (int j = 0; j < D; j++) {
      printf("%f, ", static_cast<float>(dgamma[j]));
    }
    printf("\n");
    printf("GPU dbeta:\n");
    for (int j = 0; j < D; j++) {
      printf("%f, ", static_cast<float>(dbeta[j]));
    }
    printf("\n");
    printf("GPU dx:\n");
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < D; j++) {
        printf("%f, ", static_cast<float>(dx[j + i * D]));
      }
      printf("\n");
    }
  }

  if (use_host) {
    DTYPE *dx_h = new DTYPE[N * D];
    float *dgamma_h = new float[D];
    float *dbeta_h = new float[D];
    LayerNormGradCPU(
        dy, cache_xmu, cache_xivar, gamma, dx_h, dgamma_h, dbeta_h, N, D);
    if (allow_print) {
      printf("CPU dgamma:\n");
      for (int j = 0; j < D; j++) {
        printf("%f, ", static_cast<float>(dgamma_h[j]));
      }
      printf("\n");
      printf("CPU dbeta:\n");
      for (int j = 0; j < D; j++) {
        printf("%f, ", static_cast<float>(dbeta_h[j]));
      }
      printf("\n");
      printf("CPU dx:\n");
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
          printf("%f, ", static_cast<float>(dx_h[j + i * D]));
        }
        printf("\n");
      }
    }

    bool is_same = true;
    for (int j = 0; j < D; j++) {
      float d_val = static_cast<float>(dgamma[j]);
      float h_val = static_cast<float>(dgamma_h[j]);
      if (abs(d_val - h_val > 0.03f)) {
        is_same = false;
        printf("Found diff: CPU=%f, GPU=%f at (%d,)\n", h_val, d_val, j);
        break;
      }
    }
    printf("Test pass (dgamma): %s\n", is_same ? "True" : "False");

    is_same = true;
    for (int j = 0; j < D; j++) {
      float d_val = static_cast<float>(dbeta[j]);
      float h_val = static_cast<float>(dbeta_h[j]);
      if (abs(d_val - h_val > 0.03f)) {
        is_same = false;
        printf("Found diff: CPU=%f, GPU=%f at (%d,)\n", h_val, d_val, j);
        break;
      }
    }
    printf("Test pass (dbeta): %s\n", is_same ? "True" : "False");

    is_same = true;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < D; j++) {
        float d_val = static_cast<float>(dx[j + i * D]);
        float h_val = static_cast<float>(dx_h[j + i * D]);
        if (abs(d_val - h_val > 0.03f)) {
          is_same = false;
          printf("Found diff: CPU=%f, GPU=%f at (%d, %d)\n", h_val, d_val, i, j);
          break;
        }
      }
    }
    printf("Test pass (dx): %s\n", is_same ? "True" : "False");

    delete[] dx_h;
    delete[] dgamma_h;
    delete[] dbeta_h;
  }

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(gamma));
  checkCUDA(cudaFree(beta));
  checkCUDA(cudaFree(y));
}
