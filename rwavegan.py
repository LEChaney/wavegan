import tensorflow as tf
import math
from tensorflow.python.keras.layers import multiply, add, Lambda
from ops import maxout, lrelu, round_to_nearest_multiple, residual_block, reshape
from ops import apply_phaseshuffle, z_to_gain_bias, conditional_batchnorm
from ops import dense, dense_sn, conv1d, conv_sn1d
from functools import partial

"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def RWaveGANGenerator(
    input_shape,
    slice_len=16384,
    nch=1,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=None,
    use_maxout=False,
    use_ortho_init=False,
    use_skip_z=False,
    use_sn=False):
  assert slice_len in [16384, 32768, 65536]

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 4)
  else:
    activation = tf.keras.layers.Activation('relu')

  z = tf.keras.layers.Input(input_shape)

  if use_batchnorm:
    if use_skip_z:
      normalization = lambda x: conditional_batchnorm(x, z, use_sn=use_sn, kernel_initializer=kernel_initializer, training=train)
    else:
      normalization = lambda x: tf.keras.layers.BatchNormalization()(x, training=train)
  else:
    if use_skip_z:
      def condition(x):
        gain, bias = z_to_gain_bias(z, x.shape[-1], use_sn=use_sn, kernel_initializer=kernel_initializer, training=train)
        return add([multiply([x, gain]), bias])
      normalization = condition
    else:
      normalization = lambda x: x

  def up_res_block(inputs, filters, stride=4):
    return residual_block(inputs, filters, kernel_len, 
                          stride=stride,
                          upsample=upsample,
                          normalization=normalization,
                          activation=activation,
                          kernel_initializer=kernel_initializer,
                          use_sn=use_sn,
                          training=train)
  
  if use_sn:
    which_dense = partial(dense_sn, kernel_initializer=kernel_initializer, training=train)
    which_conv = partial(conv_sn1d, kernel_initializer=kernel_initializer, training=train)
  else:
    which_dense = partial(dense, kernel_initializer=kernel_initializer)
    which_conv = partial(conv1d, kernel_initializer=kernel_initializer)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  dim_mul = 16 if slice_len == 16384 else 32
  with tf.variable_scope('z_project'):
    output = which_dense(output, 4 * 4 * dim * dim_mul)
    output = reshape(output, [16, dim * dim_mul])
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
    output = normalization(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = which_conv(output, nch, kernel_len, strides=1, padding='same')
    output = tf.keras.layers.Activation('tanh')(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    # if slice_len == 16384:
    #   assert len(update_ops) == 10
    # else:
    #   assert len(update_ops) == 12
    with tf.control_dependencies(update_ops):
      output = Lambda(lambda x: tf.identity(x))(output)

  return tf.keras.Model(inputs=z, outputs=output)


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def RWaveGANDiscriminator(
    input_shape,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    labels=None,
    nlabels=1,
    use_maxout=False,
    use_ortho_init=False,
    use_sn=False):
  slice_len = int(input_shape[0])

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 4)
  else:
    activation = lrelu

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.BatchNormalization()(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x
  
  def down_res_block(inputs, filters, stride=4):
    return residual_block(inputs, filters, kernel_len,
                          stride=stride,
                          normalization=batchnorm,
                          phaseshuffle=phaseshuffle,
                          activation=activation,
                          kernel_initializer=kernel_initializer,
                          use_sn=use_sn,
                          training=True)

  def down_res_block_no_ph(inputs, filters, stride=4):
    return residual_block(inputs, filters, kernel_len,
                          stride=stride,
                          normalization=batchnorm,
                          phaseshuffle=lambda x: x,
                          activation=activation,
                          kernel_initializer=kernel_initializer,
                          use_sn=use_sn,
                          training=True)
  
  if use_sn:
    which_dense = partial(dense_sn, kernel_initializer=kernel_initializer, training=True)
    which_conv = partial(conv_sn1d, kernel_initializer=kernel_initializer, training=True)
  else:
    which_dense = partial(dense, kernel_initializer=kernel_initializer)
    which_conv = partial(conv1d, kernel_initializer=kernel_initializer)

  inputs = tf.keras.layers.Input(input_shape)

  # From audio layer
  # [16384, nch] -> [16384, 32]
  output = inputs
  with tf.variable_scope('from_audio'):
    output = which_conv(output, dim // 2, kernel_len, strides=1, padding='same')

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
  output = tf.keras.layers.Flatten()(output)

  # Connect to single logit
  with tf.variable_scope('output'):
    if labels is not None:
      output = which_dense(output, nlabels)
      def lookup_y_hat(x):
        indices = tf.range(tf.shape(x)[0])
        return tf.gather_nd(x, tf.stack([indices, labels], -1))
      output = Lambda(lookup_y_hat)(output)
    else:
      output = Lambda(lambda x: which_dense(x, 1)[:, 0])(output)

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return tf.keras.Model(inputs=inputs, outputs=output)
