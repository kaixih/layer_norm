import numpy as np
import tensorflow as tf

epsilon = 0.001

input_shape = (3, 2, 4)
feature_axis = (1, 2)
human_readable = False

# Data preparation
ndims = len(input_shape)
nelems = np.prod(input_shape)
feature_shape=[]
for axis in feature_axis:
  feature_shape.append(input_shape[axis])
if human_readable:
  data = tf.cast(tf.reshape(tf.range(nelems), shape=input_shape), dtype=tf.float32)
  gamma = tf.ones(shape=feature_shape)
  beta = tf.zeros(shape=feature_shape)
else:
  data = tf.random.normal(shape=input_shape)
  gamma = tf.random.normal(shape=feature_shape)
  beta = tf.random.normal(shape=feature_shape)

# Keras Solution (Reference):
layer = tf.keras.layers.LayerNormalization(axis=feature_axis)
layer.build(input_shape=input_shape)
layer.set_weights([gamma, beta])

with tf.GradientTape() as tape:
  tape.watch(data)
  output = layer(data)
  loss = tf.reduce_sum(output)
dx, dy, (dgamma, dbeta) = tape.gradient(loss, [data, output, layer.variables])
print("Keras y:", output)
print("Keras dy:", dy)
print("Keras dx:", dx)
print("Keras dgamma:", dgamma)
print("Keras dbeta:", dbeta)

# Numpy Solution:
def forward(x, g, b, feature_axis):
  ndims = len(x.shape)
  mean = np.mean(x, axis=feature_axis, keepdims=True)
  var = np.var(x, axis=feature_axis, keepdims=True)
  xmu = x - mean
  xivar = np.sqrt(var + epsilon)
  x_normalized = xmu / xivar

  # Broadcasting only necessary for norm when the axis is not just the last
  # dimension.
  broadcast_shape = [1] * ndims
  for dim in feature_axis:
    broadcast_shape[dim] = input_shape[dim]

  g = np.reshape(g, broadcast_shape)
  b = np.reshape(b, broadcast_shape)
  y = x_normalized * g + b

  cache = {}
  cache["xivar"] = xivar
  cache["xmu"] = xmu
  cache["gamma"] = g

  return y, cache

def backward(dy, cache, feature_axis):
  ndims = len(dy.shape)
  batch_axis = []
  for dim in range(ndims):
    if dim not in feature_axis:
      batch_axis.append(dim)
  batch_axis = tuple(batch_axis)
  D = 1
  for dim in feature_axis:
    D *= input_shape[dim]

  xivar = cache["xivar"]
  xmu = cache["xmu"]
  g = cache["gamma"]

  dl_di = dy * g / xivar
  di_dx = 1.

  dl_dvar = np.sum(dy * g * xmu * (-0.5) * xivar**(-3),
                   axis=feature_axis, keepdims=True)
  dvar_dx = 2 * xmu / D

  dl_dmu = np.sum(-1. * dy * g / xivar, axis=feature_axis, keepdims=True) + \
           np.sum(dl_dvar * (-2. / D) * xmu, axis=feature_axis, keepdims=True)
  dmu_dx = 1. / D

  dx = dl_di * di_dx + dl_dvar * dvar_dx + dl_dmu * dmu_dx
  return dx, dgamma, dbeta


y, cache = forward(data.numpy(), gamma.numpy(), beta.numpy(), feature_axis)
dx, dgamma, dbeta = backward(dy.numpy(), cache, feature_axis)
print("--------------")
print("Numpy y:\n", y)
print("Numpy dy:\n", dy.numpy())
print("Numpy dx:\n", dx)
print("Numpy dgamma:\n", dgamma)
print("Numpy dbeta:\n", dbeta)



