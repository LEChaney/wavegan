import tensorflow as tf


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def nn_upsample(inputs, stride=4):
  '''
  Upsamples an audio clip using nearest neighbor upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('nn_upsample'):
    w = tf.shape(inputs)[1]
    output = tf.expand_dims(inputs, axis=1)
    output = tf.image.resize_nearest_neighbor(output, [1, w * stride])
    output = output[:, 0]
    return output


def avg_downsample(inputs, kernel_size=9, stride=4):
  with tf.variable_scope('avg_downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=kernel_size, strides=stride, padding='same')


def residual_block(inputs, filters, kernel_size=9, stride=1, activation=lrelu, normalization=lambda x: x, phaseshuffle = lambda x: x):
  '''
  Args:
    inputs: 
    filters:
    kernel_size: default 9
    stride: default 1
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: lambda: x
  '''
  with tf.variable_scope(None, 'res_block'):
    hidden_filters = min(inputs.shape[2], filters)

    shortcut = inputs
    if shortcut.shape[2] != filters:
      with tf.variable_scope('learned_shortcut'):
        shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same', use_bias=False)
        
    # Convolution layers
    code = inputs
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      code = tf.layers.conv1d(code, hidden_filters, kernel_size, strides=1, padding='same')
    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      code = tf.layers.conv1d(code, filters, kernel_size, strides=stride, padding='same')

    # Add shortcut connection
    code = shortcut + 0.1 * code # see https://github.com/LMescheder/GAN_stability/issues/11 for 0.1
    return code


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def RWaveGANGenerator(
    z,
    slice_len=16384,
    nch=1,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False):
  assert slice_len in [16384, 32768, 65536]
  size_scale = 16 * slice_len // 16384
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, size_scale * dim * 16)
    output = tf.reshape(output, [batch_size, size_scale, dim * 16])
    output = batchnorm(output)

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm)

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = nn_upsample(output, 4)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm)

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = nn_upsample(output, 4)
    output = residual_block(output, dim * 8, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 8, kernel_len, normalization=batchnorm)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = nn_upsample(output, 4)
    output = residual_block(output, dim * 4, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 4, kernel_len, normalization=batchnorm)
    
  # Layer 4
  # [4096, 128] -> [16384, 64]
  with tf.variable_scope('upconv_4'):
    output = nn_upsample(output, 4)
    output = residual_block(output, dim * 2, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 2, kernel_len, normalization=batchnorm)

  # Layer 5
  # [16384, 64] -> [65536, nch]
  with tf.variable_scope('upconv_5'):
    output = nn_upsample(output, 4)
    output = residual_block(output, dim * 1, kernel_len, normalization=batchnorm)
    output = residual_block(output, dim * 1, kernel_len, normalization=batchnorm)

  with tf.variable_scope('to_audio'):
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.layers.conv1d(output, nch, kernel_len, padding='SAME')
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


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def RWaveGANDiscriminator(
    x,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0):
  batch_size = tf.shape(x)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  output = x
  with tf.variable_scope('from_audio'):
    output = tf.layers.conv1d(output, dim * 1, kernel_len, padding='same')

  # Layer 0
  # [16384, 1] -> [4096, 64]
  with tf.variable_scope('downconv_0'):
    output = residual_block(output, dim * 1, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 2, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = avg_downsample(output, kernel_len, stride=4)
    output = residual_block(output, dim * 2, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 4, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = avg_downsample(output, kernel_len, stride=4)
    output = residual_block(output, dim * 4, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 8, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = avg_downsample(output, kernel_len, stride=4)
    output = residual_block(output, dim * 8, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = avg_downsample(output, kernel_len, stride=4)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Layer 5
  # [64, 1024] -> [16, 2048]
  with tf.variable_scope('downconv_5'):
    output = avg_downsample(output, kernel_len, stride=4)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)
    output = residual_block(output, dim * 16, kernel_len, normalization=batchnorm, phaseshuffle=phaseshuffle)

  # Connect to single logit
  with tf.variable_scope('output'):
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.reshape(output, [batch_size, -1])
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
