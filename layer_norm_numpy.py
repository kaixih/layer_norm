import numpy as np
from layer_norm_tf import layer_norm_tf

def layer_norm_np(x, gamma, beta, epsilon):
  assert len(x.shape) == 2
  D_axis = (1, )

  mean = np.mean(x, axis=D_axis, keepdims=True)
  var = np.var(x, axis=D_axis, keepdims=True)

  x_mean = x - mean
  ivar = 1. / np.sqrt(var + epsilon)

  x_normalized = x_mean * ivar
  y = x_normalized * gamma + beta

  cache = {}
  cache["ivar"] = ivar
  cache["x_mean"] = x_mean

  return y, cache


def layer_norm_grad_np(dy, gamma, cache):
  N_axis = (0, )
  D_axis = (1, )

  D = 1
  for dim in D_axis:
    D *= dy.shape[dim]

  ivar = cache["ivar"]
  x_mean = cache["x_mean"]

  dgamma = np.sum(dy * x_mean * ivar, axis=N_axis)
  dbeta = np.sum(dy, axis=N_axis)

  dl_di = dy * gamma * ivar
  di_dx = 1.

  dl_dvar = np.sum(dy * gamma * x_mean * (-0.5) * (ivar**3), axis=D_axis,
                   keepdims=True)
  dvar_dx = 2. * x_mean / D

  dl_dmean = np.sum(-1. * dy * gamma * ivar, axis=D_axis, keepdims=True) + \
             np.sum(dl_dvar * (-2. / D) * x_mean, axis=D_axis, keepdims=True)
  dmean_dx = 1. / D

  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmean * dmean_dx
  return dgamma, dbeta, dx


def check_close(ref, x, msg):
  assert ref.shape == x.shape
  input_shape = ref.shape
  print(f"Checking {msg}...", end='')

  if not np.allclose(ref, x, rtol=1e-3, atol=1e-3):
    ind = np.argmin(np.isclose(ref, x, rtol=1e-3, atol=1e-3))
    ind = np.unravel_index(ind, input_shape)
    print(f"\nError at {ind}: ref={ref[ind]}, np={x[ind]}")
  else:
    print("Pass")


def evaluate_np(input_shape):
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

  y_np, cache = layer_norm_np(x, gamma, beta, epsilon)
  dgamma_np, dbeta_np, dx_np = layer_norm_grad_np(dy, gamma, cache)

  check_close(y, y_np, "y")
  check_close(dgamma, dgamma_np, "dgamma")
  check_close(dbeta, dbeta_np, "dbeta")
  check_close(dx, dx_np, "dx")

#input_shapes = [
#    (10, 10000000),
#    (100, 1000000),
#    (1000, 100000),
#    (10000, 10000),
#    (100000, 1000),
#    (1000000, 100),
#    (10000000, 10),
#  ]
#for input_shape in input_shapes:
#  evaluate_np(input_shape)



def unit_test():
  dy = (np.array([2, 9, -4, 5, 8, 7, 2, 9, -4, 5, 8, 7])
            .reshape(2, 6).astype(np.float))
  gamma = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0]).astype(np.float)
  x = (np.array([1, 7, 4, -3, -11, 13, 1, 7, 4, -3, -11, 13])
           .reshape(2, 6).astype(np.float))

  mean = np.array([1.83, 1.83]).astype(np.float).reshape(2, 1)
  ivar = np.array([0.13, 0.13]).astype(np.float).reshape(2, 1)
  cache={}
  cache['x_mean'] = x - mean
  cache['ivar'] = ivar
  dgamma, dbeta, dx = layer_norm_grad_np(dy, gamma, cache)
  print("Numpy dgamma:\n", dgamma)
  print("Numpy dbeta:\n", dbeta)
  print("Numpy dx:\n", dx)

#unit_test()
