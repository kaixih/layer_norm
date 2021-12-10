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
const int kWarpSize = 32;

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
  checkCUDA(cudaDeviceSynchronize());
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
      if (fabs(d_val - h_val) > 0.03f) {
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
struct MeanOp {
  int D;
  __device__ U Compute(const T *x, const int& row, const int& col) const {
    return GetAs<T, U>(x, row * D + col);
  }
  __device__ U Finalize(const U& sum) const {
    return sum / D;
  }
};

template<typename T, typename U>
struct IvarOp {
  const U *cache_mean;
  int D;
  U epsilon;
  __device__ U Compute(const T *x, const int& row, const int& col,
                       const U& mean) const {
    U curr = GetAs<T, U>(x, row * D + col);
    return (curr - mean) * (curr - mean);
  }
  __device__ U Compute(const T *x, const int& row, const int& col) const {
    return Compute(x, row, col, cache_mean[row]);
  }
  __device__ U Finalize(const U& sum) const {
    return rsqrt(sum / D + epsilon);
  }
};

template<typename T, typename U>
struct DvarOp {
  const U *gamma;
  const T *x;
  const U *cache_ivar;
  const U *cache_mean;
  int D;
  __device__ U Compute(const T *dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return curr * gamma[col] * (x[row * D + col] - cache_mean[row]) * (-0.5) *
           (cache_ivar[row] * cache_ivar[row] * cache_ivar[row]);
  }
  __device__ U Finalize(const U& sum) const {
    return sum;
  }
};

template<typename T, typename U>
struct DmeanOp {
  const U *gamma;
  const T *x;
  const U *cache_ivar;
  const U *cache_mean;
  const U *dl_dvars;
  int D;
  __device__ U Compute(const T *dy, const int& row, const int& col,
                       const U& dl_dvar) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    return -1. * curr * gamma[col] * cache_ivar[row] + dl_dvar *
           (-2. / D) * (x[row * D + col] - cache_mean[row]);
  }
  __device__ U Compute(const T *dy, const int& row, const int& col) const {
    return Compute(dy, row, col, dl_dvars[row]);
  }
  __device__ U Finalize(const U& sum) const {
    return sum;
  }
};

template<typename T, typename U>
struct DxOp {
  const T *x;
  const U *cache_mean;
  const U *cache_ivar;
  const U *gamma;
  const U *dl_dvars;
  const U *dl_dmus;
  int D;
  __device__ T Compute(const T *dy, const int& row, const int& col) const {
    U curr = GetAs<T, U>(dy, row * D + col);
    U dl_di = curr * gamma[col] * cache_ivar[row];
    U di_dx = 1.;
    U dvar_dx = 2. * (x[row * D + col] - cache_mean[row]) / D;
    U dmu_dx = 1. / D;
    U dl_dx = dl_di * di_dx + dl_dvars[row] * dvar_dx + dl_dmus[row] * dmu_dx;
    return static_cast<T>(dl_dx);
  }
};

template<typename T, typename U>
struct YOp {
  const U *cache_mean;
  const U *cache_ivar;
  const U *gamma;
  const U *beta;
  int D;
  __device__ T Compute(const T *x, const int& row, const int& col) const {
    U mean = cache_mean[row];
    U ivar = cache_ivar[row];
    U curr = GetAs<T, U>(x, row * D + col);
    return static_cast<T>((curr - mean) * ivar * gamma[col] + beta[col]);
  }
};


template<typename T, typename U, typename Op>
__global__ void LayerNormRowReduceInToTemp(
    const T* __restrict__ x, const int N, const int D,
    U* __restrict__ temp, Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int row_offset = threadIdx.x + blockIdx.x * blockDim.x;
  
  for (int row_idx = blockIdx.y; row_idx < N; row_idx += gridDim.y) {
    U partial_sum = 0;
    for (int i = row_offset; i < D; i += gridDim.x * blockDim.x) {
      partial_sum += op.Compute(x, row_idx, i);
    }
    U sum = BlockReduce(temp_storage).Sum(partial_sum);
    if (threadIdx.x == 0) {
      temp[row_idx * gridDim.x + blockIdx.x] = sum;
    }
  }
}

template<typename U, typename Op>
__global__ void LayerNormRowReduceTempToOut(
    const U* __restrict__ temp, const int N, const int cols,
    U* __restrict__ cache, Op op) {
  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum = 0;
    for (int i = threadIdx.x; i < cols; i += kBlockSize) {
      partial_sum += temp[k * cols + i];
    }

    U sum = BlockReduce(temp_storage).Sum(partial_sum);
  
    if (threadIdx.x == 0) {
      cache[k] = op.Finalize(sum);
    }
  }
}

template<typename T, typename Op>
__global__ void LayerNormUpdate(const T* __restrict__ in, const int N,
                                const int D, T * out, Op op) {

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid >= N * D) return;

  const int col = tid % D;
  const int row = tid / D;
  out[tid] = op.Compute(in, row, col);
}



template<typename T, typename U>
void LayerNormGPU(const T* x, const U* gamma, const U* beta, const U epsilon,
                  const int N, const int D, T* y, U* cache_ivar,
                  U* cache_mean) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // const int blocks_per_row = kBlockSize * 100;
  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 100;
  bool use_single_block =
      D <= min_num_blocks * kBlockSize * min_workload_per_thread;
  // Only be used in the multiple blocks per row case.
  const int blocks_per_row = div_up(D, kBlockSize * min_workload_per_thread);

  float* temp_ivar;
  float* temp_sum;
  // if (D > kBlockSize * blocks_per_row) {
  if (!use_single_block) {
    PrepareAlloc(&temp_ivar, N * blocks_per_row, false, false);
    PrepareAlloc(&temp_sum, N * blocks_per_row, false, false);
  }

  cudaEventRecord(start);
  MeanOp<U, T> mean_ops{D};
  IvarOp<U, T> ivar_ops{cache_mean, D, 0.001f};
  // if (D <= kBlockSize * blocks_per_row) {
  if (use_single_block) {
    LayerNormRowReduceInToOut<<<N, kBlockSize>>>(
        x, N, D, cache_mean, cache_ivar, mean_ops, ivar_ops);
  } else {
    int max_num_blocks = 1000000; // to be changed.
    dim3 threads(kBlockSize, 1, 1);
    dim3 blocks(blocks_per_row, min(N, max_num_blocks / blocks_per_row), 1);
    printf("AAA blocks_per_row: %d\n", blocks_per_row);

    // Strategy: For long rows, we launch n*x blocks in total, where each x
    // blocks handle a row. The results will be stored in a temp memory, and its
    // size is N*x. Then, we launch n*1 blocks in total to reduce the temp
    // memory and each block will handle one row in temp.
    LayerNormRowReduceInToTemp<<<blocks, threads>>>(
        x, N, D, temp_sum, mean_ops);
    LayerNormRowReduceTempToOut<<<N, threads>>>(
        temp_sum, N, blocks_per_row, cache_mean, mean_ops);

    LayerNormRowReduceInToTemp<<<blocks, threads>>>(
        x, N, D, temp_ivar, ivar_ops);
    LayerNormRowReduceTempToOut<<<N, threads>>>(
        temp_ivar, N, blocks_per_row, cache_ivar, ivar_ops);
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (y) p1: %f ms\n", milliseconds);

  cudaEventRecord(start);
  YOp<T, U> y_ops{cache_mean, cache_ivar, gamma, beta, D};
  LayerNormUpdate<<<div_up(N * D, kBlockSize), kBlockSize>>>(x, N, D, y, y_ops);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (y) p2: %f ms\n", milliseconds);

  if (!use_single_block) {
    checkCUDA(cudaFree(temp_ivar));
    checkCUDA(cudaFree(temp_sum));
  }
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
  
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U dl_dvar = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      U curr = GetAs<T, U>(dy, k * D + i);
      dl_dvar += curr * gamma[i] * (x[k * D + i] - cache_mean[k]) * (-0.5) *
                 (cache_ivar[k] * cache_ivar[k] * cache_ivar[k]);
    }

    dl_dvar = BlockReduce(temp_storage.reduce).Sum(dl_dvar);

    if (tid == 0) {
      temp_storage.broadcast[0] = dl_dvar;
      dl_dvars[k] = dl_dvar;
    }
    __syncthreads();
    dl_dvar = temp_storage.broadcast[0];

    U dl_dmu = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      U curr = GetAs<T, U>(dy, k * D + i);
      dl_dmu += -1. * curr * gamma[i] * cache_ivar[k] + dl_dvar * (-2. / D) *
                (x[k * D + i] - cache_mean[k]);
    }

    dl_dmu = BlockReduce(temp_storage.reduce).Sum(dl_dmu);

    if (tid == 0) {
      dl_dmus[k] = dl_dmu;
    }
  }
}

template<typename T, typename U, typename Op1, typename Op2>
__global__ void LayerNormRowReduceInToOut(const T* __restrict__ in,
                                          const int N, const int D,
                                          U* out1, U *out2, Op1 op1, Op2 op2) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<U, kBlockSize> BlockReduce;
  __shared__ union {
      typename BlockReduce::TempStorage reduce;
      U broadcast[1];
  } temp_storage;
  
  for (int k = blockIdx.x; k < N; k += gridDim.x) {
    U partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op1.Compute(in, k, i);
    }

    U sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      temp_storage.broadcast[0] = op1.Finalize(sum);
      out1[k] = op1.Finalize(sum);
    }
    __syncthreads();
    sum = temp_storage.broadcast[0];

    partial_sum = 0;
    for (int i = tid; i < D; i += kBlockSize) {
      partial_sum += op2.Compute(in, k, i, sum);
    }

    sum = BlockReduce(temp_storage.reduce).Sum(partial_sum);

    if (tid == 0) {
      out2[k] = op2.Finalize(sum);
    }
  }
}


template<typename T, typename U>
void LayerNormGradGPU(const T* dy, const T* x, const U* cache_mean,
                      const U* cache_ivar, const U* gamma, const int N,
                      const int D, T* dx, U* dgamma, U* dbeta) {

  U* temp_1;
  U* temp_2;
  PrepareAlloc(&temp_1, N, false, false);
  PrepareAlloc(&temp_2, N, false, false);

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

  // const int blocks_per_row = kBlockSize * 50;
  const int min_num_blocks = kWarpSize;
  const int min_workload_per_thread = 50;
  bool use_single_block =
      D <= min_num_blocks * kBlockSize * min_workload_per_thread;
  // Only be used in the multiple blocks per row case.
  const int blocks_per_row = div_up(D, kBlockSize * min_workload_per_thread);

  float* temp_dl_dvars;
  float* temp_dl_dmus;
  // if (D > kBlockSize * blocks_per_row) {
  if (!use_single_block) {
    PrepareAlloc(&temp_dl_dvars, N * blocks_per_row, false, false);
    PrepareAlloc(&temp_dl_dmus, N * blocks_per_row, false, false);
  }

  cudaEventRecord(start);
  DvarOp<U, T> dl_dvar_ops{gamma, x, cache_ivar, cache_mean, D};
  DmeanOp<U, T> dl_dmu_ops{gamma, x, cache_ivar, cache_mean, temp_1, D};
  // if (D <= kBlockSize * blocks_per_row) {
  if (use_single_block) {
    LayerNormRowReduceInToOut<<<N, kBlockSize>>>(
        dy, N, D, temp_1, temp_2, dl_dvar_ops, dl_dmu_ops);
  } else {
    int max_num_blocks = 1000000; // to be changed.
    dim3 threads(kBlockSize, 1, 1);
    dim3 blocks(blocks_per_row, min(N, max_num_blocks / blocks_per_row), 1);
    printf("AAA blocks_per_row: %d\n", blocks_per_row);

    LayerNormRowReduceInToTemp<<<blocks, threads>>>(
        dy, N, D, temp_dl_dvars, dl_dvar_ops);
    LayerNormRowReduceTempToOut<<<N, threads>>>(
        temp_dl_dvars, N, blocks_per_row, temp_1, dl_dvar_ops);

    LayerNormRowReduceInToTemp<<<blocks, threads>>>(
        dy, N, D, temp_dl_dmus, dl_dmu_ops);
    LayerNormRowReduceTempToOut<<<N, threads>>>(
        temp_dl_dmus, N, blocks_per_row, temp_2, dl_dmu_ops);
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dx) p1: %f ms\n", milliseconds);

  cudaEventRecord(start);
  DxOp<T, U> dx_ops{x, cache_mean, cache_ivar, gamma, temp_1, temp_2, D};
  LayerNormUpdate<<<div_up(N * D, kBlockSize), kBlockSize>>>(
      dy, N, D, dx, dx_ops);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU time (dx) p2: %f ms\n", milliseconds);

  checkCUDA(cudaFree(temp_1));
  checkCUDA(cudaFree(temp_2));
  if (!use_single_block) {
    checkCUDA(cudaFree(temp_dl_dvars));
    checkCUDA(cudaFree(temp_dl_dmus));
  }
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

int main(int argc, char** argv) {

  /** Parameters and Knobs **/
  int N = 10000;
  int D = 10000;
  if (argc == 3) {
    N = atoi(argv[1]);
    D = atoi(argv[2]);
  }
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

  PrepareAlloc(&dy, N * D, use_host, human_readable, 99, 1);
  PrepareAlloc(&dx, N * D, use_host, human_readable);
  PrepareAlloc(&dgamma, D, use_host, human_readable);
  PrepareAlloc(&dbeta, D, use_host, human_readable);


  LayerNormGradGPU(dy, x, cache_mean, cache_ivar, gamma, N, D, dx, dgamma,
                   dbeta);
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
  // ---- Backward Done Here ----

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
}
