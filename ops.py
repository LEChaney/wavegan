import tensorflow as tf
import warnings
from functools import partial 


NO_OPS = 'NO_OPS'
SPECTRAL_NORM_UPDATE_OPS = 'spectral_norm_update_ops'


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar


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


def get_embed_table_sn(
    n_labels, 
    embedding_size, 
    table_name='embed_table',
    kernel_initializer=None,
    scope=None):
  with tf.variable_scope('embed_sn'):
    if kernel_initializer is None:
      kernel_initializer = tf.initializers.glorot_uniform
    W = tf.get_variable(table_name + '_unnorm', shape=[n_labels, embedding_size], initializer=kernel_initializer, dtype=tf.float32)

    # Get spectral normalized weight tensor
    W_bar = spectral_normed_weight(W, update_collection=SPECTRAL_NORM_UPDATE_OPS)

    return tf.identity(W_bar, name=table_name)


def embed_sn(
    y,
    n_labels,
    embedding_size,
    table_name='embed_table',
    kernel_initializer=None):
  with tf.variable_scope('embed_sn'):
    if kernel_initializer is None:
      kernel_initializer = tf.initializers.glorot_uniform
    W = tf.get_variable(table_name + '_unnorm', shape=[n_labels, embedding_size], initializer=kernel_initializer, dtype=tf.float32)

    # Get spectral normalized weight tensor
    W_bar = spectral_normed_weight(W, update_collection=SPECTRAL_NORM_UPDATE_OPS)

    # Perform layer op
    yembed = tf.nn.embedding_lookup(W_bar, y)
      
    return yembed


def dense_sn(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None):
  with tf.variable_scope('dense_sn'):
    # Init weight tensor
    if kernel_initializer is None:
      kernel_initializer = tf.initializers.glorot_uniform
    W = tf.get_variable('W', shape=[inputs.shape.as_list()[-1], units], initializer=kernel_initializer, dtype=tf.float32)

    # Get spectral normalized weight tensor
    W_bar = spectral_normed_weight(W, update_collection=SPECTRAL_NORM_UPDATE_OPS)

    # Perform layer op
    output = tf.matmul(inputs, W_bar)

    # Add bias
    if use_bias:
      output += tf.get_variable('b', initializer=tf.zeros_initializer(), shape=[output.shape.as_list()[-1]], dtype=tf.float32)
      
    return output


def conv1d_sn(
    inputs,
    filters,
    kernel_size,
    strides=4,
    padding='same',
    use_bias=True,
    kernel_initializer=None):
  with tf.variable_scope('conv1d_sn'):
    # Init weight tensor
    if kernel_initializer is None:
      kernel_initializer = tf.initializers.glorot_uniform
    W = tf.get_variable('W', shape=[kernel_size, inputs.shape.as_list()[-1], filters], initializer=kernel_initializer, dtype=tf.float32)

    # Get spectral normalized weight tensor
    W_bar = spectral_normed_weight(W, update_collection=SPECTRAL_NORM_UPDATE_OPS)

    # Perform layer op
    output = tf.nn.conv1d(inputs, W_bar, stride=strides, padding=padding.upper())

    # Add bias
    if use_bias:
      output += tf.get_variable('b', initializer=tf.zeros_initializer(), shape=[output.shape.as_list()[-1]], dtype=tf.float32)

    return output

def conv2d_transpose_sn(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='same',
    use_bias=True,
    kernel_initializer=None):
  with tf.variable_scope('conv2d_t_sn'):
    # Init weight tensor
    if kernel_initializer is None:
      kernel_initializer = tf.initializers.glorot_uniform
    W = tf.get_variable('W', shape=list(kernel_size) + [filters, inputs.shape.as_list()[-1]], initializer=kernel_initializer, dtype=tf.float32)

    # Get spectral normalized weight tensor
    W_bar = spectral_normed_weight(W, update_collection=SPECTRAL_NORM_UPDATE_OPS)

    # Perform layer op
    input_shape = inputs.shape.as_list()
    output_shape = tf.stack([tf.shape(inputs)[0], input_shape[1] * strides[0], input_shape[2] * strides[1], filters])
    strides = [1] + list(strides) + [1]
    output = tf.nn.conv2d_transpose(inputs, W_bar, output_shape=output_shape, strides=strides, padding=padding.upper())

    # Add bias
    if use_bias:
      output += tf.get_variable('b', initializer=tf.zeros_initializer(), shape=[output.shape.as_list()[-1]], dtype=tf.float32)

    return output


def conv1d_transpose(
    inputs,
    filters,
    kernel_size,
    strides=4,
    padding='same',
    upsample='zeros',
    use_bias=True,
    kernel_initializer=None,
    use_sn=False):
  which_conv2d_tranpose = conv2d_transpose_sn if use_sn else tf.layers.conv2d_transpose
  which_conv1d = conv1d_sn if use_sn else tf.layers.conv1d
  
  if upsample == 'zeros':
    return which_conv2d_tranpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_size),
        strides=(1, strides),
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)[:, 0]
  elif upsample == 'nn':
    x = nn_upsample(inputs, strides)

    return which_conv1d(
        x,
        filters,
        kernel_size,
        1,
        padding='same',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)
  else:
    raise NotImplementedError


def conv1d_transpose_sn(
    inputs,
    filters,
    kernel_size,
    strides=4,
    padding='same',
    upsample='zeros',
    use_bias=True,
    kernel_initializer=None):
  return conv1d_transpose(inputs, filters, kernel_size, strides, padding, upsample, use_bias, kernel_initializer, use_sn=True)


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


def z_to_gain_bias(z, out_features, kernel_initializer=tf.initializers.orthogonal, use_sn=False):
  if use_sn:
    which_dense = partial(dense_sn, use_bias=False, kernel_initializer=kernel_initializer)
  else:
    which_dense = partial(tf.layers.dense, use_bias=False, kernel_initializer=kernel_initializer)

  with tf.variable_scope('z_to_gain'):
    gain = 1 + which_dense(z, out_features)
    gain = tf.reshape(gain, [tf.shape(gain)[0], 1, -1])
  with tf.variable_scope('z_to_bias'):
    bias = which_dense(z, out_features)
    bias = tf.reshape(bias, [tf.shape(bias)[0], 1, -1])
  return gain, bias


def conditional_batchnorm(inputs, z, training=False, kernel_initializer=tf.initializers.orthogonal, use_sn=False):
  gain, bias = z_to_gain_bias(z, inputs.shape.as_list()[-1], kernel_initializer=kernel_initializer, use_sn=use_sn)
  return gain * tf.layers.batch_normalization(inputs, training=training, center=False, scale=False) + bias


def residual_block(inputs, 
                   filters, 
                   kernel_size=9, 
                   stride=1, 
                   upsample=None, 
                   activate_inputs=True,
                   activation=lrelu,
                   normalization=lambda x: x,
                   phaseshuffle=lambda x: x,
                   wscale=1,
                   kernel_initializer=tf.initializers.orthogonal,
                   use_sn=False):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
  '''
  with tf.variable_scope(None, 'res_block'):
    if use_sn:
      which_conv    = partial(conv1d_sn, kernel_initializer=kernel_initializer)
      which_up_conv = partial(conv1d_transpose_sn, upsample=upsample, kernel_initializer=kernel_initializer)
    else:
      which_conv    = partial(tf.layers.conv1d, kernel_initializer=kernel_initializer)
      which_up_conv = partial(conv1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer)

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
        extra_features = which_conv(shortcut, out_features - in_features,
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
        code = which_up_conv(code, internal_filters, kernel_size, strides=stride, padding='same')
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
                     wscale=1,
                     kernel_initializer=tf.initializers.orthogonal,
                     use_sn=False):
  '''
  Args:
    inputs: 
    filters: Number of output features
    kernel_size: Default 9
    stride: Default 1
    upsample: Either 'zeros' (default) or 'nn' for nearest neighbor upsample
    activation: Activation function to use default lrelu
    normalization: Normalization function to use, default: identity function
  '''
  with tf.variable_scope(None, 'res_block'):
    if use_sn:
      which_conv    = partial(conv1d_sn, kernel_initializer=kernel_initializer)
      which_up_conv = partial(conv1d_transpose_sn, upsample=upsample, kernel_initializer=kernel_initializer)
    else:
      which_conv    = partial(tf.layers.conv1d, kernel_initializer=kernel_initializer)
      which_up_conv = partial(conv1d_transpose, upsample=upsample, kernel_initializer=kernel_initializer)

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
      code = which_conv(code, internal_filters_0, kernel_size=1, strides=1, padding='same')

    # Convolutions
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = phaseshuffle(code)
      if is_upsampling:
        code = which_up_conv(code, internal_filters_0, kernel_size, strides=stride, padding='same')
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
    code = shortcut + wscale * code
    return code
