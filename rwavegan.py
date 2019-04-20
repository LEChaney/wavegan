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
    
    # Number of output features from first convolution layer
    hidden_filters = min(inputs.shape[2], filters)

    # Parameters for Fixup Initialization: https://arxiv.org/pdf/1901.09321.pdf
    act0_bias  = tf.get_variable('act0_bias' , [1, 1, inputs.shape[2]], initializer=tf.zeros_initializer(), trainable=True)
    act1_bias  = tf.get_variable('act1_bias' , [1, 1, hidden_filters ], initializer=tf.zeros_initializer(), trainable=True)
    conv0_bias = tf.get_variable('conv0_bias', [1, 1, inputs.shape[2]], initializer=tf.zeros_initializer(), trainable=True)
    conv1_bias = tf.get_variable('conv1_bias', [1, 1, hidden_filters ], initializer=tf.zeros_initializer(), trainable=True)
    conv1_mul  = tf.get_variable('conv1_mul' , [1, 1, filters        ], initializer=tf.ones_initializer() , trainable=True)
    fixup_weight_scale = 1 / math.sqrt(num_resblocks)

    code = inputs
    # Convolution layers
    with tf.variable_scope('conv_0'):
      code  = normalization(code)
      code += act0_bias
      code  = activation(code)  # Pre-Activation
      code += conv0_bias
      code  = fixup_weight_scale * tf.layers.conv1d(code, hidden_filters, kernel_size, 
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=False)
    with tf.variable_scope('conv_1'):
      code  = normalization(code)
      code += act1_bias
      code  = activation(code)  # Pre-Activation
      code += conv1_bias
      if is_upsampling:
        code = conv1d_transpose(code, filters, kernel_size, 
                                stride=stride, 
                                padding='same', 
                                upsample=upsample, 
                                use_bias=False, 
                                kernel_initializer=tf.zeros_initializer())
      else:
        code = tf.layers.conv1d(code, filters, kernel_size,
                                strides=stride,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=tf.zeros_initializer())
      code *= conv1_mul

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
  num_resblocks = 10

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

  def up_res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len, 
                          stride=4,
                          upsample=upsample,
                          num_resblocks=num_resblocks,
                          normalization=batchnorm)
  def res_block(inputs, filters):
    return residual_block(inputs, filters, kernel_len,
                          num_resblocks=num_resblocks, 
                          normalization=batchnorm)

  # Layer 0
  # [16, 1024] -> [16, 1024]
  # with tf.variable_scope('block_layer_0'):
  #   output = res_block(output, dim * 8)
  #   output = res_block(output, dim * 8)

  # Layer 1
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('block_layer_0'):
    output = up_res_block(output, dim * 8)
    output =    res_block(output, dim * 8)

  # Layer 2
  # [64, 512] -> [256, 256]
  with tf.variable_scope('block_layer_1'):
    output = up_res_block(output, dim * 4)
    output =    res_block(output, dim * 4)

  # Layer 3
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('block_layer_2'):
    output = up_res_block(output, dim * 2)
    output =    res_block(output, dim * 2)

  # Layer 4
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('block_layer_3'):
    output = up_res_block(output, dim * 1)
    output =    res_block(output, dim * 1)
    
  # # Layer 5
  # # [4096, 64] -> [16384, 32]
  with tf.variable_scope('block_layer_4'):
    output = up_res_block(output, dim // 2)
    output =    res_block(output, dim // 2)

  # To audio layer
  # [16384, 32] -> [16384, 1]
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
  nch = x.shape[2]
  num_resblocks = 10

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

  # From audio layer
  output = x
  with tf.variable_scope('from_audio'):
    output = tf.layers.conv1d(output, dim // 2, kernel_len, padding='same')

  # Layer 0
  # [16384, 32] -> [4096, 64]
  with tf.variable_scope('block_layer_0'):
    output =      res_block(output, dim // 2)
    output = down_res_block(output, dim  * 1)
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
    output = phaseshuffle(output)

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
