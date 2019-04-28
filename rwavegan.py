import tensorflow as tf
import math


def round_down_to_multiple(x, divisor):
  return math.floor(x/divisor)*divisor


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def maxout(inputs):
  return tf.contrib.layers.maxout(inputs, inputs.shape.as_list()[-1] // 2)


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


def residual_block(inputs, 
                   filters, 
                   kernel_size=9, 
                   stride=1, 
                   upsample=None, 
                   activate_inputs=True,
                   activation=lrelu,
                   normalization=lambda x: x,
                   phaseshuffle=lambda x: x):
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
    in_features = inputs.shape.as_list()[2]
    out_features =  filters
    internal_filters = min(out_features, in_features)

    # Default shortcut
    shortcut = inputs

    # Downsample shortcut before resizing feature dimension (saves computation)
    if stride > 1 and not is_upsampling:
      shortcut = avg_downsample(shortcut, stride)
    
    # Drop or concat to match number of output features
    if in_features < out_features:
      with tf.variable_scope('expand_shortcut'):
        extra_features = tf.layers.conv1d(shortcut, out_features - in_features,
                                          kernel_size=1,
                                          strides=1,
                                          padding='valid')
        shortcut = tf.concat([shortcut, extra_features], 2)
    elif in_features > out_features:
      with tf.variable_scope('drop_shortcut'):
        shortcut = shortcut[:, :, :out_features]

    # Upsample shortcut after resizing feature dimension (saves computation)
    if stride > 1 and is_upsampling:
      shortcut = nn_upsample(shortcut, stride)

    # Convolutions
    code = inputs
    with tf.variable_scope('conv_0'):
      if activate_inputs:
        code = normalization(code)
        code = activation(code)  # Pre-Activation
        code = phaseshuffle(code)
      if is_upsampling:
        code = conv1d_transpose(code, internal_filters, kernel_size, strides=stride, padding='same')
      else:
        code = tf.layers.conv1d(code, internal_filters, kernel_size, strides=1, padding='same')

    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = tf.layers.conv1d(code, out_features, kernel_size, strides=1, padding='same')
      else:
        code = tf.layers.conv1d(code, out_features, kernel_size, strides=stride, padding='same')

    # Add shortcut connection
    code = shortcut + code
    return code


def bottleneck_block(inputs, 
                     filters, 
                     kernel_size=9, 
                     stride=1, 
                     upsample=None, 
                     activate_inputs=True, 
                     activation=lrelu, 
                     normalization=lambda x: x,
                     phaseshuffle=lambda x: x):
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

    # Downsample shortcut before resizing feature dimension (saves computation)
    if stride > 1 and not is_upsampling:
      shortcut = avg_downsample(shortcut, stride)
    
    in_features = shortcut.shape[2]
    # Drop or concat to match number of output features
    if in_features < filters:
      with tf.variable_scope('expand_shortcut'):
        extra_features = tf.layers.conv1d(shortcut, filters - in_features,
                                          kernel_size=1,
                                          strides=1,
                                          padding='valid')
        shortcut = tf.concat([shortcut, extra_features], 2)
    elif in_features > filters:
      with tf.variable_scope('drop_shortcut'):
        shortcut = shortcut[:, :, :filters]

    # Upsample shortcut after resizing feature dimension (saves computation)
    if stride > 1 and is_upsampling:
      shortcut = nn_upsample(shortcut, stride)
    
    # Feature compression
    with tf.variable_scope('compress_f'):
      code = inputs
      if activate_inputs:
        code = normalization(code)
        code = activation(code)
        code = phaseshuffle(code)
      code = tf.layers.conv1d(code, internal_filters_0, kernel_size=1, strides=1, padding='same')

    # Convolutions
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = conv1d_transpose(code, internal_filters_0, kernel_size, strides=stride, padding='same')
      else:
        code = tf.layers.conv1d(code, internal_filters_0, kernel_size, strides=1, padding='same')

    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = tf.layers.conv1d(code, internal_filters_1, kernel_size, strides=1, padding='same')
      else:
        code = tf.layers.conv1d(code, internal_filters_1, kernel_size, strides=stride, padding='same')

    # Feature expansion
    with tf.variable_scope('expand_f'):
      code = normalization(code)
      code = activation(code)
      code = phaseshuffle(code)
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
    train=False,
    yembed=None,
    use_maxout=False):
  assert slice_len in [16384, 32768, 65536]
  batch_size = tf.shape(z)[0]

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_down_to_multiple(dim * math.sqrt(2), 4)
  else:
    activation = lrelu

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x
  
  def up_res_block(inputs, filters, stride=4):
    return residual_block(inputs, filters, kernel_len, 
                          stride=stride,
                          upsample=upsample,
                          normalization=batchnorm,
                          activation=activation)

  # Conditioning input
  output = z
  if yembed is not None:
    output = tf.concat([z, yembed], 1)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  dim_mul = 16 if slice_len == 16384 else 32
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * dim_mul)
    output = tf.reshape(output, [batch_size, 16, dim * dim_mul])
  dim_mul //= 2

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = up_res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = up_res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = up_res_block(output, dim * dim_mul)
  dim_mul //= 2

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = up_res_block(output, dim * dim_mul)

  if slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, 32]
    with tf.variable_scope('upconv_4'):
      output = up_res_block(output, dim // 2)

  elif slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = up_res_block(output, dim)

    # Layer 5
    # [16384, 64] -> [32768, 32]
    with tf.variable_scope('upconv_5'):
      output = up_res_block(output, dim // 2, stride=2)

  elif slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = up_res_block(output, dim)

    # Layer 5
    # [16384, 64] -> [65536, nch]
    with tf.variable_scope('upconv_5'):
      output = up_res_block(output, dim // 2)

  # To audio layer
  # [16384, 32] -> [16384, nch]
  with tf.variable_scope('to_audio'):
    output = batchnorm(output)
    output = activation(output)
    output = tf.layers.conv1d(output, nch, kernel_len, strides=1, padding='same')
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
    phaseshuffle_rad=0,
    labels=None,
    nlabels=1,
    use_maxout=False):
  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_down_to_multiple(dim * math.sqrt(2), 4)
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
  
  def down_res_block(inputs, filters, stride=4, activate_inputs=True):
    return residual_block(inputs, filters, kernel_len,
                          stride=stride,
                          normalization=batchnorm,
                          phaseshuffle=phaseshuffle,
                          activation=activation)

  def down_res_block_no_ph(inputs, filters, stride=4):
    return residual_block(inputs, filters, kernel_len,
                          stride=stride,
                          normalization=batchnorm,
                          phaseshuffle=lambda x: x,
                          activation=activation)

  # From audio layer
  # [16384, nch] -> [16384, 32]
  output = x
  with tf.variable_scope('from_audio'):
    output = tf.layers.conv1d(output, dim // 2, kernel_len, strides=1, padding='same')

  # Layer 0
  # [16384, 32] -> [4096, 64]
  with tf.variable_scope('downconv_0'):
    output = down_res_block(output, dim)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = down_res_block(output, dim * 2)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = down_res_block(output, dim * 4)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = down_res_block(output, dim * 8)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = down_res_block(output, dim * 16)

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = down_res_block_no_ph(output, dim * 32, stride=2)
  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = down_res_block_no_ph(output, dim * 32)
  
  # Activate final layer
  output = batchnorm(output)
  output = activation(output)

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  with tf.variable_scope('output'):
    if labels is not None:
      output = tf.layers.dense(output, nlabels)
      indices = tf.range(tf.shape(output)[0])
      output = tf.gather_nd(output, tf.stack([indices, labels], -1))
    else:
      output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
