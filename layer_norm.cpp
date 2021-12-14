#include <cmath>
#include <iostream>

template <typename T, typename U>
U GetAs(const T* in, int offset) {
  return static_cast<U>(in[offset]);
}

template <typename T, typename U>
void LayerNormCPU(const T* x, const U* gamma, const U* beta, const int N,
                  const int D, const U epsilon, T* y) {
  for (int j = 0; j < N; j++) {
    U mean, ivar;
    U sum = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum += curr;
    }
    mean = sum / D;

    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = 1.0 / sqrt(sum_ivar / D + epsilon);

    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      y[j * D + i] = static_cast<T>((curr - mean) * ivar * gamma[i] + beta[i]);
    }
  }
}

template <typename T, typename U>
void LayerNormGradCPU(const T* dy, const T* x, const U* gamma, const int N,
                      const int D, const U epsilon, U* dgamma, U* dbeta,
                      T* dx) {
  U* cache_mean = new U[N];
  U* cache_ivar = new U[N];
  for (int j = 0; j < N; j++) {
    U mean, ivar;
    U sum = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum += curr;
    }
    mean = sum / D;

    U sum_ivar = 0;
    for (int i = 0; i < D; i++) {
      U curr = GetAs<T, U>(x, j * D + i);
      sum_ivar += (curr - mean) * (curr - mean);
    }
    ivar = 1.0 / sqrt(sum_ivar / D + epsilon);

    cache_mean[j] = mean;
    cache_ivar[j] = ivar;
  }

  // Compute dgamma, dbeta.
  for (int i = 0; i < D; i++) {
    dgamma[i] = 0;
    dbeta[i] = 0;
    for (int j = 0; j < N; j++) {
      U dy_curr = static_cast<U>(dy[j * D + i]);
      dgamma[i] += dy_curr * (x[j * D + i] - cache_mean[j]) * cache_ivar[j];
      dbeta[i] += dy_curr;
    }
  }

  // Compute dx.
  for (int i = 0; i < N; i++) {
    U dl_dvar = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dvar += curr * gamma[j] * (x[i * D + j] - cache_mean[i]) * (-0.5) *
                 (cache_ivar[i] * cache_ivar[i] * cache_ivar[i]);
    }

    U dl_dmean = 0;
    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      dl_dmean += -1. * curr * gamma[j] * cache_ivar[i];
      dl_dmean += dl_dvar * (-2. / D) * (x[i * D + j] - cache_mean[i]);
    }

    for (int j = 0; j < D; j++) {
      U curr = static_cast<U>(dy[i * D + j]);
      U dl_di = curr * gamma[j] * cache_ivar[i];
      U di_dx = 1.;

      // dl_dvar is above.
      U dvar_dx = 2. * (x[i * D + j] - cache_mean[i]) / D;

      // dl_dmean is above.
      U dmean_dx = 1. / D;

      U dl_dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx;
      dx[i * D + j] = static_cast<T>(dl_dx);
    }
  }

  delete[] cache_mean;
  delete[] cache_ivar;
}

extern "C" {
void layer_norm(const float* x, const float* gamma, const float* beta,
                const int N, const int D, const float epsilon, float* y) {
  LayerNormCPU(x, gamma, beta, N, D, epsilon, y);
}

void layer_norm_grad(const float* dy, const float* x, const float* gamma,
                     const int N, const int D, const float epsilon, float* dx,
                     float* dgamma, float* dbeta) {
  LayerNormGradCPU(dy, x, gamma, N, D, epsilon, dgamma, dbeta, dx);
}
}
