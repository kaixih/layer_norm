import numpy as np

from ctypes import *
from layer_norm_tf import layer_norm_tf

lib = cdll.LoadLibrary('./libln.so')

def check_close(ref, x, msg, rtol, atol):
  assert ref.shape == x.shape
  input_shape = ref.shape
  print(f"Checking {msg}...", end='')

  if not np.allclose(ref, x, rtol=rtol, atol=atol):
    ind = np.argmin(np.isclose(ref, x, rtol=rtol, atol=atol))
    ind = np.unravel_index(ind, input_shape)
    print(f"\nError at {ind}: ref={ref[ind]}, cpp={x[ind]}")
  else:
    print("Pass")


def evaluate_cpp(input_shape, rtol=1e-3, atol=1e-3):
  print(f"Evaluating {input_shape}...")
  assert len(input_shape) == 2
  epsilon = 0.001
  dtype = np.float32

  np.random.seed(12)
  x = np.random.normal(size=input_shape).astype(dtype)
  gamma = np.random.normal(size=input_shape[1]).astype(dtype)
  beta = np.random.normal(size=input_shape[1]).astype(dtype)
  dy = np.ones(shape=input_shape, dtype=dtype)

  y, dgamma, dbeta, dx = layer_norm_tf(x, gamma, beta, epsilon)

  y_cpp = np.empty_like(x)
  dx_cpp = np.empty_like(x)
  dgamma_cpp = np.empty_like(gamma)
  dbeta_cpp = np.empty_like(beta)

  lib.layer_norm(
      x.ctypes.data_as(POINTER(c_float)),
      gamma.ctypes.data_as(POINTER(c_float)),
      beta.ctypes.data_as(POINTER(c_float)),
      c_int(input_shape[0]),
      c_int(input_shape[1]),
      c_float(epsilon),
      y_cpp.ctypes.data_as(POINTER(c_float)))
  lib.layer_norm_grad(
      dy.ctypes.data_as(POINTER(c_float)),
      x.ctypes.data_as(POINTER(c_float)),
      gamma.ctypes.data_as(POINTER(c_float)),
      c_int(input_shape[0]),
      c_int(input_shape[1]),
      c_float(epsilon),
      dx_cpp.ctypes.data_as(POINTER(c_float)),
      dgamma_cpp.ctypes.data_as(POINTER(c_float)),
      dbeta_cpp.ctypes.data_as(POINTER(c_float)))

  check_close(y, y_cpp, "y", rtol, atol)
  check_close(dgamma, dgamma_cpp, "dgamma", rtol, atol)
  check_close(dbeta, dbeta_cpp, "dbeta", rtol, atol)
  check_close(dx, dx_cpp, "dx", rtol, atol)


input_shapes = [
    (10, 10000000),
    (100, 1000000),
    (1000, 100000),
    (10000, 10000),
    (100000, 1000),
    (1000000, 100),
    (10000000, 10),
  ]
for input_shape in input_shapes:
  if input_shape == (10, 10000000):
    evaluate_cpp(input_shape, 1e-2, 1e-1)
    continue
  evaluate_cpp(input_shape)


