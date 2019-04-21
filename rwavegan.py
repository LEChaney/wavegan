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
    kernel_initializer=tf.initializers.he_uniform()):
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


def residual_block(
    inputs, 
    filters, 
    kernel_size=9, 
    stride=1, 
    upsample=None, 
    activate_inputs=True, 
    activation=lrelu, 
    normalization=lambda x: x,
    kernel_initializer=tf.initializers.he_uniform(),
    output_layer_initializer=tf.initializers.he_uniform()):
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
    
    # Project to match output shape
    if (shortcut.shape[2] != filters or stride > 1) and not is_upsampling:
      with tf.variable_scope('proj_shortcut'):
        shortcut = tf.layers.conv1d(shortcut, filters,
                                    kernel_size=1,
                                    strides=stride,
                                    padding='valid',
                                    kernel_initializer=output_layer_initializer)
    elif is_upsampling:
      with tf.variable_scope('proj_shortcut'):
        shortcut = tf.layers.conv1d(shortcut, filters,
                                    kernel_size=1,
                                    strides=1,
                                    padding='valid',
                                    kernel_initializer=output_layer_initializer)
        shortcut = nn_upsample(shortcut, stride)
    
    # Feature compression
    with tf.variable_scope('compress_f'):
      code = inputs
      if activate_inputs:
        code = normalization(code)
        code = activation(code)
      code = tf.layers.conv1d(code, internal_filters_0, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer)

    # Convolutions
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      if is_upsampling:
        code = conv1d_transpose(code, internal_filters_0, kernel_size, stride=stride, padding='same', kernel_initializer=kernel_initializer)
      else:
        code = tf.layers.conv1d(code, internal_filters_0, kernel_size, strides=stride, padding='same', kernel_initializer=kernel_initializer)
    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = tf.layers.conv1d(code, internal_filters_1, kernel_size, strides=1, padding='same', kernel_initializer=kernel_initializer)

    # Feature expansion
    with tf.variable_scope('expand_f'):
      code = normalization(code)
      code = activation(code)
      code = tf.layers.conv1d(code, filters, kernel_size=1, strides=1, padding='same', kernel_initializer=output_layer_initializer)

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

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  def up_res_block(inputs, filters, output_layer_initializer=tf.initializers.he_uniform()):
    return residual_block(inputs, filters, kernel_len, 
                          stride=4,
                          upsample=upsample,
                          normalization=batchnorm,
                          output_layer_initializer=output_layer_initializer)
  def res_block(inputs, filters, output_layer_initializer=tf.initializers.he_uniform()):
    return residual_block(inputs, filters, kernel_len,
                          normalization=batchnorm,
                          output_layer_initializer=output_layer_initializer)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, size_scale * dim * 16,
                             kernel_initializer=tf.initializers.he_uniform())
    output = tf.reshape(output, [batch_size, size_scale, dim * 16])

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('block_layer_0'):
    output = up_res_block(output, dim * 8)
    output =    res_block(output, dim * 8)

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('block_layer_1'):
    output = up_res_block(output, dim * 4)
    output =    res_block(output, dim * 4)

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('block_layer_2'):
    output = up_res_block(output, dim * 2)
    output =    res_block(output, dim * 2)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('block_layer_3'):
    output = up_res_block(output, dim * 1)
    output =    res_block(output, dim * 1)
    
  # # Layer 4
  # # [4096, 64] -> [16384, nch]
  with tf.variable_scope('block_layer_4'):
    output = up_res_block(output, nch)
    output =    res_block(output, nch,
                          output_layer_initializer=tf.initializers.glorot_uniform())
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
  def down_res_block(inputs, filters, activate_inputs=True):
    return residual_block(inputs, filters, kernel_len,
                          stride=4,
                          normalization=batchnorm,
                          activate_inputs=activate_inputs)

  output = x

  # Layer 0
  # [16384, nch] -> [4096, 64]
  with tf.variable_scope('block_layer_0'):
    output =      res_block(output, nch, activate_inputs=False)
    output = down_res_block(output, dim)
    output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('block_layer_1'):
    output =      res_block(output, dim * 1)
    output = down_res_block(output, dim * 2)
    output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('block_layer_2'):
    output =      res_block(output, dim * 2)
    output = down_res_block(output, dim * 4)
    output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('block_layer_3'):
    output =      res_block(output, dim * 4)
    output = down_res_block(output, dim * 8)
    output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('block_layer_4'):
    output =      res_block(output, dim * 8)
    output = down_res_block(output, dim * 16)

  # Connect to single logit
  with tf.variable_scope('output'):
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.reshape(output, [batch_size, -1])
    output = tf.layers.dense(output, 1,
                             kernel_initializer=tf.initializers.he_uniform())[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
