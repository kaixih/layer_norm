import tensorflow as tf

def layer_norm_tf(x, gamma, beta, epsilon):
  x = tf.convert_to_tensor(x)
  input_shape = x.shape
  assert len(input_shape) == 2

  feature_axis = (1, )
  layer = tf.keras.layers.LayerNormalization(axis=feature_axis, epsilon=epsilon)
  layer.build(input_shape=input_shape)
  layer.set_weights([gamma, beta])

  def train_step(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = layer(x)
      loss = tf.reduce_sum(y)
    dx, dy, (dgamma, dbeta) = tape.gradient(loss, [x, y, layer.variables])
    return y, dgamma, dbeta, dx, dy

  y, dgamma, dbeta, dx, dy = train_step(x)

  dy_is_one = tf.reduce_all(dy == 1.)
  assert dy_is_one.numpy() == True
  return y, dgamma, dbeta, dx


