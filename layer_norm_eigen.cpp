#include <time.h>

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename T>
void IsClose2DHost(const T* x, const T* y, int N, int D, std::string msg,
                   float atol = 1e-3, float rtol = 1e-3);

template <typename T>
void Print2DHost(const T* x, int N, int D, std::string msg);

template <typename T, typename U>
void LayerNormCPU(const T* x, const U* gamma, const U* beta, const int N,
                  const int D, const U epsilon, T* y);

template <typename T, typename U>
void LayerNormGradCPU(const T* dy, const T* x, const U* gamma, const int N,
                      const int D, const U epsilon, U* dgamma, U* dbeta, T* dx);

#define DTYPE float

template <typename T>
void InitAlloc(T* x, int size, int init = -1) {
  srand(12);
  for (int i = 0; i < size; i++) {
    if (init != -1) {
      x[i] = init;
    } else {
      x[i] = static_cast<T>(static_cast<float>(rand()) / RAND_MAX);
    }
  }
}

template <typename T>
void SetEigenTensor(T* out, T* in, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}

template <typename T, typename U>
void LayerNormEigen(const Eigen::Tensor<T, 2, Eigen::RowMajor>& in,
                    const Eigen::Tensor<U, 1, Eigen::RowMajor>& scale,
                    const Eigen::Tensor<U, 1, Eigen::RowMajor>& offset,
                    const int N, const int D, const U epsilon,
                    Eigen::Tensor<T, 2, Eigen::RowMajor>& y) {
  Eigen::array<int, 1> reduce_dims({1});
  Eigen::DSizes<Eigen::Index, 2> N_by_one(N, 1);
  Eigen::DSizes<Eigen::Index, 2> one_by_D(1, D);
  Eigen::array<int, 2> bcast_D({1, D});
  Eigen::array<int, 2> bcast_N({N, 1});
  Eigen::Tensor<float, 1, Eigen::RowMajor> mean(N);
  Eigen::Tensor<float, 1, Eigen::RowMajor> variance(N);

  float D_inv = 1.0f / D;
  auto x = in.template cast<U>();
  mean = x.sum(reduce_dims) * D_inv;

  auto x_centered = x - mean.reshape(N_by_one).broadcast(bcast_D);

  variance = x_centered.square().sum(reduce_dims) * D_inv;

  auto scaling_factor =
      (variance + epsilon).rsqrt().eval().reshape(N_by_one).broadcast(bcast_D) *
      scale.reshape(one_by_D).broadcast(bcast_N);
  auto x_scaled = x_centered * scaling_factor;

  auto x_shifted = (x_scaled + offset.reshape(one_by_D).broadcast(bcast_N));

  y = x_shifted.template cast<T>();
}

template <typename T, typename U>
void LayerNormGradEigen(const Eigen::Tensor<T, 2, Eigen::RowMajor>& dy,
                        const Eigen::Tensor<T, 2, Eigen::RowMajor>& in,
                        const Eigen::Tensor<U, 1, Eigen::RowMajor>& scale,
                        const int N, const int D, const U epsilon,
                        Eigen::Tensor<U, 1, Eigen::RowMajor>& dscale,
                        Eigen::Tensor<U, 1, Eigen::RowMajor>& doffset,
                        Eigen::Tensor<T, 2, Eigen::RowMajor>& dx) {
  Eigen::array<int, 1> reduce_D({1});
  Eigen::array<int, 1> reduce_N({0});
  Eigen::DSizes<Eigen::Index, 2> N_by_one(N, 1);
  Eigen::DSizes<Eigen::Index, 2> one_by_D(1, D);
  Eigen::array<int, 2> bcast_D({1, D});
  Eigen::array<int, 2> bcast_N({N, 1});
  Eigen::Tensor<float, 1, Eigen::RowMajor> mean(N);
  Eigen::Tensor<float, 2, Eigen::RowMajor> ivar(N, D);

  float D_inv = 1.0f / D;
  auto x = in.template cast<U>();
  mean = x.sum(reduce_D) * D_inv;

  auto x_centered = (x - mean.reshape(N_by_one).broadcast(bcast_D)).eval();

  auto variance = x_centered.square().sum(reduce_D) * D_inv;

  ivar =
      (variance + epsilon).rsqrt().eval().reshape(N_by_one).broadcast(bcast_D);

  dscale = (dy * x_centered * ivar).sum(reduce_N);
  doffset = dy.sum(reduce_N);

  // Compute dl_di: dy * scale * ivar
  auto dl_di = (dy * scale.reshape(one_by_D).broadcast(bcast_N) * ivar).eval();
  U di_dx = 1.;

  // Compute dl_dvar: (dy * scale * x_centered * -0.5 * ivar^3).sum(reduce_D)
  auto dl_dvar =
      ((dl_di * x_centered * (-0.5f) * ivar * ivar).sum(reduce_D)).eval();
  auto dvar_dx = (2.f * x_centered * D_inv).eval();

  // Compute dl_mean: (-1 * dy * scale * ivar).sum(reduce_D) + (dl_dvar * -2 / D
  // * x_centered).sum(reduce_D)
  auto dl_dmean = (-1.f * dl_di).sum(reduce_D).eval() +
                  (dl_dvar.reshape(N_by_one).broadcast(bcast_D) * (-2.f) *
                   D_inv * x_centered)
                      .sum(reduce_D)
                      .eval();
  U dmean_dx = 1.f * D_inv;

  auto out = dl_di * di_dx +
             dl_dvar.reshape(N_by_one).broadcast(bcast_D) * dvar_dx +
             dl_dmean.reshape(N_by_one).broadcast(bcast_D) * dmean_dx;
  dx = out.template cast<T>();
}

int main(int argc, char** argv) {
  int N = 10000;
  int D = 10000;
  if (argc >= 3) {
    N = atoi(argv[1]);
    D = atoi(argv[2]);
  }
  bool allow_print = false;

  DTYPE* x_data = new DTYPE[N * D];
  float* gamma_data = new float[D];
  float* beta_data = new float[D];
  InitAlloc(x_data, N * D);
  InitAlloc(gamma_data, D);
  InitAlloc(beta_data, D);

  const float epsilon = 0.001f;
  Eigen::Tensor<DTYPE, 2, Eigen::RowMajor> x(N, D);
  Eigen::Tensor<DTYPE, 2, Eigen::RowMajor> y(N, D);
  Eigen::Tensor<float, 1, Eigen::RowMajor> scale(D);
  Eigen::Tensor<float, 1, Eigen::RowMajor> offset(D);
  SetEigenTensor(x.data(), x_data, N * D);
  SetEigenTensor(scale.data(), gamma_data, D);
  SetEigenTensor(offset.data(), beta_data, D);
  double time_spent = 0.0;
  clock_t begin = clock();

  LayerNormEigen(x, scale, offset, N, D, epsilon, y);

  clock_t end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("Eigen time: %f ms\n", time_spent);

  if (allow_print) {
    std::cout << "Eigen y:" << std::endl;
    std::cout << y << std::endl;
  }

  DTYPE* y_data = new DTYPE[N * D];

  time_spent = 0.0;
  begin = clock();

  LayerNormCPU(x_data, gamma_data, beta_data, N, D, epsilon, y_data);

  end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("CPU time: %f ms\n", time_spent);

  if (allow_print) {
    Print2DHost(y_data, N, D, "CPU y:");
  }

  IsClose2DHost(y_data, (float*)y.data(), N, D, "y");

  Eigen::Tensor<DTYPE, 2, Eigen::RowMajor> dy(N, D);
  dy.setConstant(1.);
  Eigen::Tensor<float, 1, Eigen::RowMajor> dscale(D);
  Eigen::Tensor<float, 1, Eigen::RowMajor> doffset(D);
  Eigen::Tensor<DTYPE, 2, Eigen::RowMajor> dx(N, D);
  time_spent = 0.0;
  begin = clock();
  LayerNormGradEigen(dy, x, scale, N, D, epsilon, dscale, doffset, dx);

  end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("Eigen Grad time: %f ms\n", time_spent);
  if (allow_print) {
    std::cout << "Eigen dgamma:" << std::endl;
    std::cout << dscale << std::endl;
    std::cout << "Eigen dbeta:" << std::endl;
    std::cout << doffset << std::endl;
    std::cout << "Eigen dx:" << std::endl;
    std::cout << dx << std::endl;
  }

  float* dgamma_data = new float[D];
  float* dbeta_data = new float[D];
  DTYPE* dx_data = new DTYPE[N * D];
  time_spent = 0.0;
  begin = clock();
  LayerNormGradCPU((DTYPE*)dy.data(), x_data, gamma_data, N, D, epsilon,
                   dgamma_data, dbeta_data, dx_data);
  end = clock();
  time_spent += (double)(end - begin) / (CLOCKS_PER_SEC / 1000);
  printf("CPU Grad time: %f ms\n", time_spent);
  if (allow_print) {
    Print2DHost(dgamma_data, 1, D, "CPU dgamma:");
    Print2DHost(dbeta_data, 1, D, "CPU dbeta:");
    Print2DHost(dx_data, N, D, "CPU dx:");
  }

  // We need larger atol and rtol mainly when N is too large. Computing dgamma
  // is essentially a reduction over N dimension.
  IsClose2DHost(dgamma_data, (float*)dscale.data(), 1, D, "dgamma", 1e-2, 1e-2);
  IsClose2DHost(dbeta_data, (float*)doffset.data(), 1, D, "dbeta");
  IsClose2DHost(dx_data, (DTYPE*)dx.data(), N, D, "dx");

  delete[] x_data;
  delete[] gamma_data;
  delete[] beta_data;
  delete[] y_data;
  delete[] dgamma_data;
  delete[] dbeta_data;
  delete[] dx_data;
}
