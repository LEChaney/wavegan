import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, multiply, add, concatenate
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2DTranspose
from functools import partial

def round_to_nearest_multiple(x, divisor):
  return round(x/divisor)*divisor


def lrelu(inputs, alpha=0.2):
  return tf.keras.layers.LeakyReLU(alpha=alpha)(inputs)


def maxout(inputs):
  def _maxout(inputs):
    tf.contrib.layers.maxout(inputs, inputs.shape.as_list()[-1] // 2)
  return Lambda(_maxout)(inputs)


def nn_upsample(inputs, stride=4):
  '''
  Upsamples an audio clip using nearest neighbor upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('nn_upsample'):
    return tf.keras.layers.UpSampling1D(size=stride)(inputs)


def avg_downsample(inputs, stride=4):
  with tf.variable_scope('avg_downsample'):
    return tf.keras.layers.AveragePooling1D(pool_size=stride, strides=stride, padding='same')(inputs)


def dense(inputs, *args, **kwargs):
  return tf.keras.layers.Dense(*args, **kwargs)(inputs)

def dense_sn(inputs, *args, training=None, **kwargs):
  return DenseSN(*args, **kwargs)(inputs, training=training)

def conv1d(inputs, *args, **kwargs):
  return tf.keras.layers.Conv1D(*args, **kwargs)(inputs)

def conv_sn1d(inputs, *args, training=None, **kwargs):
  return ConvSN1D(*args, **kwargs)(inputs, training=training)

def conv2d_transpose(inputs, *args, **kwargs):
  return tf.keras.layers.Conv2DTranspose(*args, **kwargs)(inputs)

def conv_sn2d_tranpose(inputs, *args, training=None, **kwargs):
  return ConvSN2DTranspose(*args, **kwargs)(inputs, training=training)

def reshape(inputs, *args, **kwargs):
  return tf.keras.layers.Reshape(*args, **kwargs)(inputs)

def conv1d_transpose(
    inputs,
    filters,
    kernel_size,
    strides=4,
    padding='same',
    upsample='zeros',
    use_bias=True,
    kernel_initializer=None,
    use_sn=False,
    training=None):
  if use_sn:
    which_conv = partial(conv_sn1d, kernel_initializer=kernel_initializer, use_bias=use_bias, training=training)
    which_conv_transpose = partial(conv_sn2d_tranpose, kernel_initializer=kernel_initializer, use_bias=use_bias, training=training)
  else:
    which_conv = partial(conv1d, kernel_initializer=kernel_initializer, use_bias=use_bias)
    which_conv_transpose = partial(conv2d_transpose, kernel_initializer=kernel_initializer, use_bias=use_bias)

  if upsample == 'zeros':
    return Lambda(lambda x: which_conv_transpose(
        Lambda(lambda x: tf.expand_dims(x, 1))(x),
        filters,
        (1, kernel_size),
        strides=(1, strides),
        padding='same')[:, 0])(inputs)
  elif upsample == 'nn':
    x = nn_upsample(inputs, strides)

    return which_conv(
        x,
        filters,
        kernel_size,
        1,
        padding='same')
  else:
    raise NotImplementedError

def conv_sn1d_transpose(*args, **kwargs):
  kwargs['use_sn'] = True
  return conv1d_transpose(*args, **kwargs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  def _apply_phaseshuffle(x):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

  return Lambda(_apply_phaseshuffle)(x)


def z_to_gain_bias(z, out_features, kernel_initializer=tf.initializers.orthogonal, use_sn=False, training=None):
  if use_sn:
    which_dense = partial(dense_sn, training=training, kernel_initializer=kernel_initializer, use_bias=False)
  else:
    which_dense = partial(dense, kernel_initializer=kernel_initializer, use_bias=False)

  with tf.variable_scope('z_to_gain'):
    gain = Lambda(lambda z: 1 + which_dense(z, out_features))(z)
    gain = reshape(gain, [1, -1])
  with tf.variable_scope('z_to_bias'):
    bias = which_dense(z, out_features)
    bias = reshape(bias, [1, -1])
  return gain, bias


def conditional_batchnorm(inputs, z, kernel_initializer=tf.initializers.orthogonal, use_sn=False, training=None):
  gain, bias = z_to_gain_bias(z, inputs.shape[-1], kernel_initializer=kernel_initializer, use_sn=use_sn, training=training)
  batch_normed = tf.layers.BatchNormalization(center=False, scale=False)(inputs, training=training)
  return add([multiply([gain, batch_normed]), bias])


def residual_block(inputs, 
                   filters, 
                   kernel_size=9, 
                   stride=1, 
                   upsample=None, 
                   activate_inputs=True,
                   activation=lrelu,
                   normalization=lambda x: x,
                   phaseshuffle=lambda x: x,
                   kernel_initializer=tf.initializers.orthogonal,
                   use_sn=False,
                   training=None):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
    kernel_initializer: The initializer to use for initializing weights
    use_sn: Whether to use spectral normalized weights
  '''
  if use_sn:
    which_conv = partial(conv_sn1d, kernel_initializer=kernel_initializer, training=training)
    which_upconv = partial(conv_sn1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer, training=training)
  else:
    which_conv = partial(conv1d, kernel_initializer=kernel_initializer)
    which_upconv = partial(conv1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer)
  
  with tf.variable_scope(None, 'res_block'):
    is_upsampling = (upsample == 'zeros' or upsample == 'nn')
    in_features = inputs.shape.as_list()[2]
    out_features = filters
    internal_filters = min(out_features, in_features)

    # Default shortcut
    shortcut = inputs

    # Downsample shortcut before resizing feature dimension (saves computation)
    if stride > 1 and not is_upsampling:
      shortcut = avg_downsample(shortcut, stride)
    
    # Drop or concat to match number of output features
    if in_features < out_features:
      with tf.variable_scope('expand_shortcut'):
        extra_features = which_conv(shortcut, out_features - in_features,
                                kernel_size=1,
                                strides=1,
                                padding='valid')
        shortcut = concatenate([shortcut, extra_features])
    elif in_features > out_features:
      with tf.variable_scope('drop_shortcut'):
        shortcut = Lambda(lambda x: x[:, :, :out_features])(shortcut)

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
        code = which_upconv(code, internal_filters, kernel_size, strides=stride, padding='same')
      else:
        code = which_conv(code, internal_filters, kernel_size, strides=1, padding='same')

    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = which_conv(code, out_features, kernel_size, strides=1, padding='same')
      else:
        code = which_conv(code, out_features, kernel_size, strides=stride, padding='same')

    # Add shortcut connection
    code = add([shortcut, code])
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
                     kernel_initializer=tf.initializers.orthogonal,
                     use_sn=False,
                     training=None):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
    kernel_initializer: The initializer to use for initializing weights
    use_sn: Whether to use spectral normalized weights
  '''
  if use_sn:
    which_conv = partial(conv_sn1d, kernel_initializer=kernel_initializer, training=training)
    which_upconv = partial(conv_sn1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer, training=training)
  else:
    which_conv = partial(conv1d, kernel_initializer=kernel_initializer)
    which_upconv = partial(conv1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer)

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
        extra_features = which_conv(shortcut, filters - in_features,
                                kernel_size=1,
                                strides=1,
                                padding='valid')
        shortcut = concatenate([shortcut, extra_features])
    elif in_features > filters:
      with tf.variable_scope('drop_shortcut'):
        shortcut = Lambda(lambda x: x[:, :, :filters])(shortcut)

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
      code = which_conv(code, internal_filters_0, kernel_size=1, strides=1, padding='same')

    # Convolutions
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = which_upconv(code, internal_filters_0, kernel_size, strides=stride, padding='same')
      else:
        code = which_conv(code, internal_filters_0, kernel_size, strides=1, padding='same')

    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = which_conv(code, internal_filters_1, kernel_size, strides=1, padding='same')
      else:
        code = which_conv(code, internal_filters_1, kernel_size, strides=stride, padding='same')

    # Feature expansion
    with tf.variable_scope('expand_f'):
      code = normalization(code)
      code = activation(code)
      code = phaseshuffle(code)
      code = which_conv(code, filters, kernel_size=1, strides=1, padding='same')

    # Add shortcut connection
    code = add([shortcut, code])
    return code
