import tensorflow as tf
import math

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


def avg_downsample(inputs, stride=4):
  with tf.variable_scope('avg_downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=stride, strides=stride, padding='same')


def conv1d_transpose(
    inputs,
    filters,
    kernel_size,
    strides=4,
    padding='same',
    upsample='zeros',
    use_bias=True,
    kernel_initializer=None):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_size),
        strides=(1, strides),
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)[:, 0]
  elif upsample == 'nn':
    x = nn_upsample(inputs, strides)

    return tf.layers.conv1d(
        x,
        filters,
        kernel_size,
        1,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)
  else:
    raise NotImplementedError


def residual_block(inputs, filters, kernel_size=9, stride=1, upsample=None, activate_inputs=True, activation=lrelu, normalization=lambda x: x):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    num_resblocks: The number of residual blocks in the network, used for scaling initialization
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
  '''
  with tf.variable_scope(None, 'res_block'):
    is_upsampling = (upsample == 'zeros' or upsample == 'nn')
    internal_filters = min(filters, inputs.shape[2])

    # Default shortcut
    shortcut = inputs

    # Resize shortcut to match output length
    if is_upsampling:
      shortcut = nn_upsample(shortcut, stride)
    elif stride > 1:
      shortcut = avg_downsample(shortcut, stride)
    
    # Project to match number of output features
    if shortcut.shape[2] != filters:
      with tf.variable_scope('proj_shortcut'):
        shortcut = tf.layers.conv1d(shortcut, filters,
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid')

    # Convolutions
    code = inputs
    with tf.variable_scope('conv_0'):
      if activate_inputs:
        code = normalization(code)
        code = activation(code)  # Pre-Activation
      if is_upsampling:
        code = conv1d_transpose(code, internal_filters, kernel_size, strides=stride, padding='same')
      else:
        code = tf.layers.conv1d(code, internal_filters, kernel_size, strides=stride, padding='same')
    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')

    # Add shortcut connection
    code = shortcut + code
    return code


def bottleneck_block(inputs, filters, kernel_size=9, stride=1, upsample=None, activate_inputs=True, activation=lrelu, normalization=lambda x: x):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    num_resblocks: The number of residual blocks in the network, used for scaling initialization
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
  '''
  with tf.variable_scope(None, 'res_block'):
    is_upsampling = (upsample == 'zeros' or upsample == 'nn')
    min_internal_filters = 2
    filters_div = 4
    internal_filters_0 = min(filters, inputs.shape[2])
    internal_filters_0 = max(internal_filters_0 // filters_div, min_internal_filters)
    internal_filters_1 = max(filters // filters_div, min_internal_filters)

    # Default shortcut
    shortcut = inputs

    # Resize shortcut to match output length
    if is_upsampling:
      shortcut = nn_upsample(shortcut, stride)
    elif stride > 1:
      shortcut = avg_downsample(shortcut, stride)
    
    # Project to match number of output features
    if shortcut.shape[2] != filters:
      with tf.variable_scope('proj_shortcut'):
        shortcut = tf.layers.conv1d(shortcut, filters,
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid')
    
    # Feature compression
    with tf.variable_scope('compress_f'):
      code = inputs
      if activate_inputs:
        code = normalization(code)
        code = activation(code)
      code = tf.layers.conv1d(code, internal_filters_0, kernel_size=1, strides=1, padding='same')

    # Convolutions
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      if is_upsampling:
        code = conv1d_transpose(code, internal_filters_0, kernel_size, strides=stride, padding='same')
      else:
        code = tf.layers.conv1d(code, internal_filters_0, kernel_size, strides=stride, padding='same')
    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = tf.layers.conv1d(code, internal_filters_1, kernel_size, strides=1, padding='same')

    # Feature expansion
    with tf.variable_scope('expand_f'):
      code = normalization(code)
      code = activation(code)
      code = tf.layers.conv1d(code, filters, kernel_size=1, strides=1, padding='same')

    # Add shortcut connection
    code = shortcut + code
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
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  def res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len,
                          normalization=batchnorm)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  dim_mul = 16 if slice_len == 16384 else 32
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * dim_mul)
    output = tf.reshape(output, [batch_size, 16, dim * dim_mul])
    output = batchnorm(output)
  output = tf.nn.relu(output)
  dim_mul //= 2

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    res_block(output, dim * dim_mul)

  if slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, nch]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample)
      res_block(output, nch)
    output = tf.nn.tanh(output)
  elif slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
      res_block(output, dim)

    # Layer 5
    # [16384, 64] -> [32768, nch]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch, kernel_len, 2, upsample=upsample)
      res_block(output, nch)
    output = tf.nn.tanh(output)
  elif slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
      res_block(output, dim)

    # Layer 5
    # [16384, 64] -> [65536, nch]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch, kernel_len, 4, upsample=upsample)
      res_block(output, nch)
    output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    # if slice_len == 16384:
    #   assert len(update_ops) == 10
    # else:
    #   assert len(update_ops) == 12
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
  slice_len = int(x.get_shape()[1])
  nch = x.shape[2]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  def res_block(inputs, filters, activate_inputs=True):
    return residual_block(inputs, filters, kernel_len,
                          normalization=batchnorm,
                          activate_inputs=activate_inputs)

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output =        res_block(output, nch, activate_inputs=False)
    output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
    output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output =        res_block(output, dim)
    output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
    output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output =        res_block(output, dim * 2)
    output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
    output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output =        res_block(output, dim * 4)
    output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
    output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output =        res_block(output, dim * 8)
    output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output =        res_block(output, dim * 16)
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 2, padding='SAME')
  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output =        res_block(output, dim * 32)
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 4, padding='SAME')
  
  # Activate final layer
  output = batchnorm(output)
  output = lrelu(output)

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
