import tensorflow as tf
import math
from ops import maxout, lrelu, round_to_nearest_multiple, apply_phaseshuffle, conv1d_transpose, conditional_batchnorm, z_to_gain_bias


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def WaveGANGenerator(
    z,
    slice_len=16384,
    nch=1,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    yembed=None,
    use_maxout=False,
    use_ortho_init=False,
    use_skip_z=False):
  assert slice_len in [16384, 32768, 65536]
  batch_size = tf.shape(z)[0]

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 2)
  else:
    activation = tf.nn.relu

  # Conditioning input
  if yembed is not None:
    z = tf.concat([z, yembed], 1)

  if use_batchnorm:
    if use_skip_z:
      normalization = lambda x: conditional_batchnorm(x, z, training=train, kernel_initializer=kernel_initializer)
    else:
      normalization = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    if use_skip_z:
      def condition(x):
        gain, bias = z_to_gain_bias(z, x.shape[-1], kernel_initializer=kernel_initializer)
        return x * gain + bias
      normalization = condition
    else:
      normalization = lambda x: x

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  dim_mul = 16 if slice_len == 16384 else 32
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * dim_mul, kernel_initializer=kernel_initializer)
    output = tf.reshape(output, [batch_size, 16, dim * dim_mul])
    output = normalization(output)
  output = activation(output)
  dim_mul //= 2

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = normalization(output)
  output = activation(output)
  dim_mul //= 2

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = normalization(output)
  output = activation(output)
  dim_mul //= 2

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = normalization(output)
  output = activation(output)
  dim_mul //= 2

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = normalization(output)
  output = activation(output)

  if slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, nch]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = tf.nn.tanh(output)
  elif slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
      output = normalization(output)
    output = activation(output)

    # Layer 5
    # [16384, 64] -> [32768, nch]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch, kernel_len, 2, upsample=upsample, kernel_initializer=kernel_initializer)
    output = tf.nn.tanh(output)
  elif slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
      output = normalization(output)
    output = activation(output)

    # Layer 5
    # [16384, 64] -> [65536, nch]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample, kernel_initializer=kernel_initializer)
    output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    if slice_len == 16384:
      assert len(update_ops) == 10
    else:
      assert len(update_ops) == 12
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    labels=None,
    nlabels=1,
    use_maxout=False,
    use_ortho_init=False):
  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 2)
  else:
    activation = lrelu

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
  output = activation(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
    output = batchnorm(output)
  output = activation(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
    output = batchnorm(output)
  output = activation(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
    output = batchnorm(output)
  output = activation(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
    output = batchnorm(output)
  output = activation(output)

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 2, padding='SAME', kernel_initializer=kernel_initializer)
      output = batchnorm(output)
    output = activation(output)
  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 4, padding='SAME', kernel_initializer=kernel_initializer)
      output = batchnorm(output)
    output = activation(output)

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  with tf.variable_scope('output'):
    if labels is not None:
      output = tf.layers.dense(output, nlabels, kernel_initializer=kernel_initializer)
      indices = tf.range(tf.shape(output)[0])
      output = tf.gather_nd(output, tf.stack([indices, labels], -1))
    else:
      output = tf.layers.dense(output, 1, kernel_initializer=kernel_initializer)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
