import tensorflow as tf

def round_to_nearest_multiple(x, divisor):
  return round(x/divisor)*divisor


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


def residual_block(inputs, 
                   filters, 
                   kernel_size=9, 
                   stride=1, 
                   upsample=None, 
                   activate_inputs=True,
                   activation=lrelu,
                   normalization=lambda x: x,
                   phaseshuffle=lambda x: x,
                   wscale=1):
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
    code = shortcut + wscale * code
    return code


def bottleneck_block(inputs, 
                     filters, 
                     kernel_size=9, 
                     stride=1, 
                     upsample=None, 
                     activate_inputs=True, 
                     activation=lrelu, 
                     normalization=lambda x: x,
                     phaseshuffle=lambda x: x,
                     wscale=1):
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
    code = shortcut + wscale * code
    return code
