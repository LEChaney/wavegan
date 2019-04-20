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
    stride=4,
    padding='same',
    upsample='zeros',
    use_bias=True,
    kernel_initializer=None):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_size),
        strides=(1, stride),
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)[:, 0]
  elif upsample == 'nn':
    x = nn_upsample(inputs, stride)

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


def residual_block(inputs, filters, kernel_size=9, stride=1, upsample=None, num_resblocks=10, activation=lrelu, normalization=lambda x: x):
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
    min_hidden_filters = 4
    hidden_filters = min(filters, inputs.shape[2])
    hidden_filters = max(hidden_filters // 4, min_hidden_filters)

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
                                      padding='same',
                                      use_bias=False)

    code = inputs
    
    # Feature compression
    with tf.variable_scope('compress_f'):
      code0 = code
      if is_upsampling:
        code0 = nn_upsample(code0, stride)
      elif stride > 1:
        code0 = avg_downsample(code0, stride)
      if code0.shape[2] != hidden_filters:
        code0 = normalization(code0)
        code0 = activation(code0)
        code0 = tf.layers.conv1d(code0, hidden_filters, kernel_size=1, strides=1, padding='same')

    # Convolutions
    with tf.variable_scope('conv_0'):
      code1 = code
      code1 = normalization(code1)
      code1 = activation(code1)  # Pre-Activation
      if is_upsampling:
        code1 = conv1d_transpose(code1, hidden_filters, kernel_size, stride=stride, padding='same')
      else:
        code1 = tf.layers.conv1d(code1, hidden_filters, kernel_size, strides=stride, padding='same')
    with tf.variable_scope('conv_1'):
      code2 = tf.concat([code0, code1], 2)
      code2 = normalization(code2)
      code2 = activation(code2)  # Pre-Activation
      code2 = tf.layers.conv1d(code2, hidden_filters, kernel_size, strides=1, padding='same')

    # Feature expansion
    with tf.variable_scope('expand_f'):
      code3 = tf.concat([code1, code2], 2)
      if code3.shape[2] != filters:
        code3 = normalization(code3)
        code3 = activation(code3)
        code3 = tf.layers.conv1d(code2, filters, kernel_size=1, strides=1, padding='same')
    
    code = code3

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
  size_scale = 16 * slice_len // 16384
  batch_size = tf.shape(z)[0]
  num_resblocks = 5

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  def up_res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len, 
                          stride=4,
                          upsample=upsample,
                          num_resblocks=num_resblocks,
                          normalization=batchnorm,
                          activation=tf.nn.relu)
  def res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len,
                          num_resblocks=num_resblocks, 
                          normalization=batchnorm,
                          activation=tf.nn.relu)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, size_scale * dim * 16)
    output = tf.reshape(output, [batch_size, size_scale, dim * 16])

  # Layer 0
  # [16, 1024] -> [16, 1024]
  # with tf.variable_scope('block_layer_0'):
  #   output = res_block(output, dim * 8)
  #   output = res_block(output, dim * 8)

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('block_layer_0'):
    output = up_res_block(output, dim * 8)
    # output =    res_block(output, dim * 8)

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('block_layer_1'):
    output = up_res_block(output, dim * 4)
    # output =    res_block(output, dim * 4)

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('block_layer_2'):
    output = up_res_block(output, dim * 2)
    # output =    res_block(output, dim * 2)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('block_layer_3'):
    output = up_res_block(output, dim * 1)
    # output =    res_block(output, dim * 1)
    
  # # Layer 4
  # # [4096, 64] -> [16384, 32]
  with tf.variable_scope('block_layer_4'):
    output = up_res_block(output, nch)
    # output =    res_block(output, nch)
    output = tf.nn.tanh(output)

  # To audio layer
  # [16384, 32] -> [16384, 1]
  # with tf.variable_scope('to_audio'):
  #   output = batchnorm(output)
  #   output = lrelu(output)
  #   output = tf.layers.conv1d(output, nch, kernel_len, padding='SAME')
    # output = tf.nn.tanh(output)

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
  nch = x.shape[2]
  num_resblocks = 5

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  def res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len,
                          num_resblocks=num_resblocks,
                          normalization=batchnorm)
  def down_res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len,
                          num_resblocks=num_resblocks,
                          stride=4,
                          normalization=batchnorm)

  output = x

  # From audio layer
  # with tf.variable_scope('from_audio'):
  #   output = tf.layers.conv1d(output, dim // 2, kernel_len, padding='same')

  # Layer 0
  # [16384, 32] -> [4096, 64]
  with tf.variable_scope('block_layer_0'):
    # output =      res_block(output, nch     )
    output = down_res_block(output, dim  * 1)
    output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('block_layer_1'):
    # output =      res_block(output, dim * 1)
    output = down_res_block(output, dim * 2)
    output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('block_layer_2'):
    # output =      res_block(output, dim * 2)
    output = down_res_block(output, dim * 4)
    output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('block_layer_3'):
    # output =      res_block(output, dim * 4)
    output = down_res_block(output, dim * 8)
    output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('block_layer_4'):
    # output =      res_block(output, dim * 8)
    output = down_res_block(output, dim * 16)
    # output = phaseshuffle(output)

  # # Layer 5
  # # [16, 1024] -> [16, 1024]
  # with tf.variable_scope('block_layer_5'):
  #   output = res_block(output, dim * 8)
  #   output = res_block(output, dim * 16)

  # Connect to single logit
  with tf.variable_scope('output'):
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.reshape(output, [batch_size, -1])
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
