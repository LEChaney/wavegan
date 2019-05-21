import tensorflow as tf
import math
from ops import maxout, lrelu, round_to_nearest_multiple, bottleneck_block, apply_phaseshuffle, conditional_batchnorm, z_to_gain_bias
from ops import dense_sn, conv1d_sn, embed_sn
from functools import partial

"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def DRWaveGANGenerator(
    z,
    slice_len=16384,
    nch=1,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    yembed=None,
    use_maxout=False,
    use_ortho_init=True,
    use_skip_z=False,
    use_spec_norm=False):
  assert slice_len in [16384, 32768, 65536]
  batch_size = tf.shape(z)[0]
  size = slice_len // 1024
  use_spec_norm = False

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 2)
  else:
    activation = tf.nn.relu

  # Conditioning input
  if yembed is not None:
    z = tf.concat([z, yembed], 1)

  if use_batchnorm:
    if use_skip_z:
      normalization = lambda x: conditional_batchnorm(x, z, training=train, kernel_initializer=kernel_initializer, use_sn=use_spec_norm)
    else:
      normalization = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    if use_skip_z:
      def condition(x):
        gain, bias = z_to_gain_bias(z, x.shape.as_list()[-1], kernel_initializer=kernel_initializer, use_sn=use_spec_norm)
        return x * gain + bias
      normalization = condition
    else:
      normalization = lambda x: x

  if use_spec_norm:
    which_dense = partial(dense_sn, kernel_initializer=kernel_initializer)
    which_conv  = partial(conv1d_sn, kernel_initializer=kernel_initializer)
  else:
    which_dense = partial(tf.layers.dense, kernel_initializer=kernel_initializer)
    which_conv  = partial(tf.layers.conv1d, kernel_initializer=kernel_initializer)

  def res_block(inputs, filters):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=1,
                            normalization=normalization,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)
  
  def up_res_block(inputs, filters, stride=4):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=stride,
                            upsample=upsample,
                            normalization=normalization,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  with tf.variable_scope('z_project'):
    output = which_dense(output, dim * 16 * size)
    output = tf.reshape(output, [batch_size, size, dim * 16])

  # Layer 0
  # [16, 1024] -> [64, 1024]
  with tf.variable_scope('res_0'):
    output =    res_block(output, dim * 16)
    output = up_res_block(output, dim * 16)

  # Layer 1
  # [64, 1024] -> [256, 512]
  with tf.variable_scope('res_1'):
    output =    res_block(output, dim * 16)
    output = up_res_block(output, dim * 8)

  # Layer 2
  # [256, 512] -> [1024, 256]
  with tf.variable_scope('res_2'):
    output =    res_block(output, dim * 8)
    output = up_res_block(output, dim * 4)

  # Layer 3
  # [1024, 256] -> [4096, 128]
  with tf.variable_scope('res_3'):
    output =    res_block(output, dim * 4)
    output = up_res_block(output, dim * 2)

  # Layer 4
  # [4096, 128] -> [16384, 64]
  with tf.variable_scope('res_4'):
    output =    res_block(output, dim * 2)
    output = up_res_block(output, dim * 1)

  # To audio layer
  # [16384, 64] -> [16384, nch]
  with tf.variable_scope('to_audio'):
    output = normalization(output)
    output = tf.nn.relu(output)
    output = which_conv(output, nch, kernel_len, strides=1, padding='same')
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


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def DRWaveGANDiscriminator(
    x,
    kernel_len=9,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    y=None,
    n_labels=1,
    use_maxout=False,
    use_ortho_init=True,
    use_spec_norm=False):
  batch_size = tf.shape(x)[0]

  # Select initialization method
  if use_ortho_init:
    kernel_initializer = tf.initializers.orthogonal
  else:
    kernel_initializer = None

  if use_maxout:
    activation = maxout
    # Because we are halving the output size of every activation.
    # This should bring the model back to the same total number of parameters.
    dim = round_to_nearest_multiple(dim * math.sqrt(2), 2)
  else:
    activation = tf.nn.relu

  if use_batchnorm:
    batchnorm = lambda x: x #tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  if use_spec_norm:
    which_dense = partial(dense_sn, kernel_initializer=kernel_initializer)
    which_conv  = partial(conv1d_sn, kernel_initializer=kernel_initializer)
  else:
    which_dense = partial(tf.layers.dense, kernel_initializer=kernel_initializer)
    which_conv  = partial(tf.layers.conv1d, kernel_initializer=kernel_initializer)

  def res_block(inputs, filters):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=1,
                            normalization=batchnorm,
                            phaseshuffle=phaseshuffle,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)

  def res_block_no_ph(inputs, filters):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=1,
                            normalization=batchnorm,
                            phaseshuffle=lambda x: x,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)
  
  def down_res_block(inputs, filters, stride=4):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=stride,
                            normalization=batchnorm,
                            phaseshuffle=phaseshuffle,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)

  def down_res_block_no_ph(inputs, filters, stride=4):
    return bottleneck_block(inputs, filters, kernel_len,
                            stride=stride,
                            normalization=batchnorm,
                            phaseshuffle=lambda x: x,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            use_sn=use_spec_norm)

  # From audio layer
  # [16384, nch] -> [16384, 64]
  output = x
  with tf.variable_scope('from_audio'):
    output = which_conv(output, dim, kernel_len, strides=1, padding='same')

  # Layer 0
  # [16384, 64] -> [4096, 128]
  with tf.variable_scope('res_0'):
    output = down_res_block_no_ph(output, dim * 2)
    output =      res_block_no_ph(output, dim * 2)

  # Layer 1
  # [4096, 128] -> [1024, 256]
  with tf.variable_scope('res_1'):
    output =  down_res_block(output, dim * 4)
    output = res_block_no_ph(output, dim * 4)

  # Layer 2
  # [1024, 256] -> [256, 512]
  with tf.variable_scope('res_2'):
    output =  down_res_block(output, dim * 8)
    output = res_block_no_ph(output, dim * 8)

  # Layer 3
  # [256, 512] -> [64, 1024]
  with tf.variable_scope('res_3'):
    output =  down_res_block(output, dim * 16)
    output = res_block_no_ph(output, dim * 16)

  # Layer 4
  # [64, 1024] -> [16, 1024]
  with tf.variable_scope('res_4'):
    output =  down_res_block(output, dim * 16)
    output = res_block_no_ph(output, dim * 16)
  
  # Activate final layer
  output = batchnorm(output)
  output = activation(output)

  # Global pooling
  # [16, 1024] -> [1024]
  with tf.variable_scope('global_pool'):
    pool = tf.reduce_sum(output, axis=1)

  # Connect to single logit
  with tf.variable_scope('output'):
    output = which_dense(pool, 1)[:, 0]

    if y is not None:
      embed_size = pool.shape.as_list()[-1]
      yembed = embed_sn(y, n_labels, embed_size, kernel_initializer=kernel_initializer)
      output += tf.reduce_sum(yembed * pool, axis=1)

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
