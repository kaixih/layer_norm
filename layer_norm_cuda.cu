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
    }
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

#define DTYPE __half

template<typename T, typename U>
void PrepareAlloc(T **x, U **gamma, U **beta, T **y, int N, int D,
                  bool use_host, bool human_readable) {
  T *buf_x = new T[N * D];
  U *buf_gamma = new U[D];
  U *buf_beta = new U[D];
  for (int i = 0; i < N * D; i++) {
    if (human_readable) {
      buf_x[i] = i;
    } else {
      buf_x[i] = static_cast<DTYPE>(static_cast<float>(rand()) / RAND_MAX);
    }
  }
  for (int i = 0; i < D; i++) {
    if (human_readable) {
      buf_gamma[i] = 1;
    } else {
      buf_gamma[i] = static_cast<float>((int)rand()) / RAND_MAX;
    }
  }
  for (int i = 0; i < D; i++) {
    if (human_readable) {
      buf_beta[i] = 0;
    } else {
      buf_beta[i] = static_cast<float>((int)rand()) / RAND_MAX;
    }
  }
  if (use_host) {
    checkCUDA(cudaMallocManaged(&(*x), N * D * sizeof(DTYPE)));
    for (int i = 0; i < N * D; i++) {
      (*x)[i] = buf_x[i];
    }
    checkCUDA(cudaMallocManaged(&(*gamma), D * sizeof(float)));
    for (int i = 0; i < D; i++) {
      (*gamma)[i] = buf_gamma[i];
    }
    checkCUDA(cudaMallocManaged(&(*beta), D * sizeof(float)));
    for (int i = 0; i < D; i++) {
      (*beta)[i] = buf_beta[i];
    }
    checkCUDA(cudaMallocManaged(&(*y), N * D * sizeof(DTYPE)));
  } else {
    checkCUDA(cudaMalloc(&(*x), N * D * sizeof(DTYPE)));
    checkCUDA(cudaMalloc(&(*gamma), D * sizeof(float)));
    checkCUDA(cudaMalloc(&(*beta), D * sizeof(float)));
    checkCUDA(cudaMalloc(&(*y), N * D * sizeof(DTYPE)));
    checkCUDA(cudaMemcpy(*x, buf_x,  N * D * sizeof(DTYPE),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(*gamma, buf_gamma,  D * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(*beta, buf_beta,  D * sizeof(float),
                         cudaMemcpyHostToDevice));
  }

  delete[] buf_x;
  delete[] buf_gamma;
  delete[] buf_beta;
}

int main() {

  /** Parameters and Knobs **/
  int N = 3;
  int D = 8;
  bool allow_print = true;
  bool human_readable = false;
  bool use_host = true;

  DTYPE* x;
  float* gamma;
  float* beta;
  DTYPE* y;

  PrepareAlloc(&x, &gamma, &beta, &y, N, D, use_host, human_readable);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  const dim3 threads(kBlockSize, 1);
  const dim3 blocks(50, 1, 1);
  LayerNormKernel<<<blocks, threads>>>(x, gamma, beta, 0.001f, y, N, D);
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

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(gamma));
  checkCUDA(cudaFree(beta));
  checkCUDA(cudaFree(y));
}
