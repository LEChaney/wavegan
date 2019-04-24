import tensorflow as tf


def lerp_clip(a, b, t): 
  return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


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
  with tf.variable_scope('downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=stride, strides=stride, padding='same')


def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    x = nn_upsample(inputs, stride)

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    raise NotImplementedError


def to_audio(inputs, out_nch, kernel_len=1, stride=1):
  '''
  Converts feature map into an audio clip.
  '''
  assert(out_nch == 1 or out_nch == 2)

  with tf.variable_scope('to_audio', reuse=tf.AUTO_REUSE):
    in_feature_maps = inputs.shape[2]

    # TODO: Add normalization
    if in_feature_maps == out_nch:
      output = inputs
    elif stride > 1:
      # Combine upsample with 'to audio' transform
      output = conv1d_transpose(inputs, filters=out_nch, kernel_width=kernel_len, stride=stride, padding='same')
    else:
      output = tf.layers.conv1d(inputs, filters=out_nch, kernel_size=kernel_len, strides=1, padding='same')
    output = tf.nn.tanh(output)
    return output


def from_audio(inputs, out_feature_maps, kernel_len=1, stride=1):
  '''
  Converts an input audio clip into a feature maps.
  '''
  with tf.variable_scope('from_audio', reuse=tf.AUTO_REUSE):
    in_nch = inputs.shape[2]
    assert(in_nch == 1 or in_nch == 2)

    # TODO: Add normalization
    # TODO: Make phaseshuffle radius adjustable via settings
    if in_nch == out_feature_maps:
      output = inputs
    else:
      output = tf.layers.conv1d(inputs, filters=out_feature_maps, kernel_size=kernel_len, strides=stride, padding='same')
      output = lrelu(output)
      output = apply_phaseshuffle(output, 2)
    return output


def up_block(inputs, audio_lod, on_amount, filters, kernel_len=25, stride=4, upsample='zeros'):
  '''
  Up Block
  '''
  nch = audio_lod.shape[2]

  mode = tf.get_variable('mode', initializer=tf.constant(-1.0), trainable=False)
  tf.summary.scalar('block_mode', mode)

  def conv_layers():
    with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
      # TODO: Add normalization
      code = inputs
      code = conv1d_transpose(code, filters, kernel_len, stride=stride, upsample=upsample)
      # code = tf.nn.relu(code)
      # code = tf.layers.conv1d(code, filters, kernel_len, strides=1, padding='SAME')
      return code

  def skip():
    with tf.control_dependencies([mode.assign(0)]):
      out_audio_lod = nn_upsample(audio_lod, stride)

      # Output code and output audio should be the same for the final up block
      if nch == filters:
        code = out_audio_lod
      else:
        code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)

      return code, out_audio_lod

  def transition():
    with tf.control_dependencies([mode.assign(1)]):
      # Blend this LOD block in over time
      out_audio_lod = to_audio(inputs, nch, kernel_len=kernel_len, stride=stride)
      out_audio_lod = lerp_clip(nn_upsample(audio_lod, stride), out_audio_lod, on_amount)
    
      # When we are transitioning, the next block is garanteed to be off (skipping) and has no
      # need for the code output from this block.
      # Output code and output audio should be the same for the final up block
      if nch == filters:
        code = out_audio_lod
      else:
        code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)

      return code, out_audio_lod

  def fully_on():
    with tf.control_dependencies([mode.assign(2)]):
      out_audio_lod = to_audio(inputs, nch, kernel_len=kernel_len, stride=stride)

      # Output code and output audio should be the same for the final up block
      if nch == filters:
        code = out_audio_lod
      else:
        # We only need to output code if the next layer is transitioning or fully on
        code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)

      return code, out_audio_lod
  
  def next_layer_transitioning():
    # Avoid going past 'fully on' state if this is the last block
    if nch == filters:
      return fully_on()

    with tf.control_dependencies([mode.assign(3)]):
      out_audio_lod = to_audio(inputs, nch, kernel_len=kernel_len, stride=stride)
      code = conv_layers()
      return code, out_audio_lod

  def next_layer_fully_on():
    # Avoid going past 'fully on' state if this is the last block
    if nch == filters:
      return fully_on()

    with tf.control_dependencies([mode.assign(4)]):
      code = conv_layers()
      
      # When the next layer is fully on we don't need to calculate an input audio lod for it
      # anymore, since it will only be using the 'code' output of this layer to calculate its
      # audio and code outputs.
      out_audio_lod = tf.zeros([tf.shape(audio_lod)[0], tf.shape(audio_lod)[1] * stride, tf.shape(audio_lod)[2]], dtype=tf.float32)

      return code, out_audio_lod

  code, out_audio_lod = tf.case([(                               on_amount <= 0,  skip),
                                 (tf.logical_and(0 <  on_amount, on_amount <  1), transition),
                                 (                      tf.equal(on_amount,   1), fully_on),
                                 (tf.logical_and(1 <  on_amount, on_amount <  2), next_layer_transitioning)],
                                default = next_layer_fully_on)

  code.set_shape([inputs.shape[0], inputs.shape[1] * stride, filters])
  out_audio_lod.set_shape([audio_lod.shape[0], audio_lod.shape[1] * stride, audio_lod.shape[2]])
  assert(audio_lod.shape[2] == 1 or audio_lod.shape[2] == 2)

  return code, out_audio_lod


def down_block(inputs, audio_lod, on_amount, filters, kernel_len=25, stride=4, last_block=False):
  '''
  Down Block
  '''
  mode = tf.get_variable('mode', initializer=tf.constant(-1.0), trainable=False)
  tf.summary.scalar('block_mode', mode)

  def conv_layers():
    with tf.variable_scope('conv_layers', reuse=tf.AUTO_REUSE):
      # Blend this LOD block in over time
      # TODO: Add normalization
      # TODO: Make phase shuffle adjustable
      code = inputs
      # code = tf.layers.conv1d(code, code.shape[2], kernel_len, strides=1, padding='SAME')
      # code = lrelu(code)
      # code = apply_phaseshuffle(code, 2)
      code = tf.layers.conv1d(code, filters, kernel_len, strides=stride, padding='SAME')
      return code

  def next_layer_fully_off():
    # Ensure we don't skip code output if we are the final layer
    if last_block:
      return next_layer_fully_on()

    with tf.control_dependencies([mode.assign(0)]):
      # Since the next layer is fully off, it will only need the output audio lod
      # from this layer. There is no need to calculate any kind of output code here.
      code = tf.zeros([inputs.shape[0], inputs.shape[1] // stride, filters], dtype=tf.float32)
      out_audio_lod = avg_downsample(audio_lod, stride)
      return code, out_audio_lod

  def next_layer_transitioning():
    # Ensure we skip audio output if we are the final layer
    if last_block:
      return next_layer_fully_on()

    with tf.control_dependencies([mode.assign(1)]):
      code = from_audio(audio_lod, filters, kernel_len, stride)
      out_audio_lod = avg_downsample(audio_lod, stride)
      return code, out_audio_lod

  def next_layer_fully_on():
    with tf.control_dependencies([mode.assign(2)]):
      code = from_audio(audio_lod, filters, kernel_len, stride)
      
      # The next layer is garanteed to be fully on, and only requires code
      # output from this layer. Audio output can be safefly disabled
      out_audio_lod = tf.zeros([inputs.shape[0], inputs.shape[1] // stride, filters], dtype=tf.float32)
      return code, out_audio_lod

  def transition():
    with tf.control_dependencies([mode.assign(3)]):
      skip_code = from_audio(audio_lod, filters, kernel_len, stride)
      code = conv_layers()
      code = lerp_clip(skip_code, code, on_amount)

      # The next layer is garanteed to be fully on, and only requires code
      # output from this layer. Audio output can be safefly disabled
      out_audio_lod = tf.zeros([audio_lod.shape[0], audio_lod.shape[1] // stride, audio_lod.shape[2]], dtype=tf.float32)
      return code, out_audio_lod

  def fully_on():
    with tf.control_dependencies([mode.assign(4)]):
      # When this layer is fully on we don't need to calculate the downsampled audio lod and
      # convert it to a code to blend between anymore. We can simply output the calculated
      # output code from the this block inputs. The next downsample block is also garanteed 
      # to be fully on, so it will also only need code output from this block.
      out_audio_lod = tf.zeros([audio_lod.shape[0], audio_lod.shape[1] // stride, audio_lod.shape[2]], dtype=tf.float32)
      code = conv_layers() 
      return code, out_audio_lod

  code, out_audio_lod = tf.case([(on_amount                                  <= -1,  next_layer_fully_off),
                                 (tf.logical_and(-1 <   on_amount, on_amount <   0), next_layer_transitioning),
                                 (                        tf.equal(on_amount,    0), next_layer_fully_on),
                                 (tf.logical_and( 0 <   on_amount, on_amount <   1), transition)],
                                default = fully_on)

  code.set_shape([inputs.shape[0], inputs.shape[1] // stride, filters])
  out_audio_lod.set_shape([audio_lod.shape[0], audio_lod.shape[1] // stride, audio_lod.shape[2]])
  assert(out_audio_lod.shape[2] == 1 or out_audio_lod.shape[2] == 2)

  return code, out_audio_lod


def n_nn_upsamples(audio, num_upsamples, stride=4):
  if num_upsamples > 0:
    return n_nn_upsamples(nn_upsample(audio, stride), num_upsamples - 1)
  else:
    return audio


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def PWaveGANGenerator(
    z,
    lod,
    slice_len=16384,
    nch=1,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    embedding=None):
  assert slice_len in [16384, 32768, 65536]
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

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

  # First layer only needs audio input while it skipping or transitioning.
  # Once the layer is fully on it no longer needs audio LOD input anymore,
  # so we can safely turn this off to save computation.
  audio_lod = tf.cond(lod < 1,
    lambda: to_audio(output, nch),
    lambda: tf.zeros([tf.shape(output)[0], tf.shape(output)[1], nch]))

  # Audio Summary
  # TODO: Allow using custom sample rate (not just hard coded to 16KHz)
  if slice_len == 16384:
    max_lod  = 5 # Use less upsamples for producing summaries when we have less upsample layers
  else:
    max_lod  = 6
  tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod), 16000, max_outputs=10, family='G_audio_lod_0')

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    on_amount = lod  # on at LOD 1
    output, audio_lod = up_block(output, audio_lod, on_amount, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2

    # Summary for LOD level
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 1), 16000, max_outputs=10, family='G_audio_lod_1')

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    on_amount = lod - 1  # on at LOD 2
    output, audio_lod = up_block(output, audio_lod, on_amount, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2

    # Summary for LOD level
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 2), 16000, max_outputs=10, family='G_audio_lod_2')

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    on_amount = lod - 2  # on at LOD 3
    output, audio_lod = up_block(output, audio_lod, on_amount, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
    output = tf.nn.relu(output)
    dim_mul //= 2

    # Summary for LOD level
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 3), 16000, max_outputs=10, family='G_audio_lod_3')

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    on_amount = lod - 3  # on at LOD 4
    output, audio_lod = up_block(output, audio_lod, on_amount, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
    output = tf.nn.relu(output)

    # Summary for LOD level
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 4), 16000, max_outputs=10, family='G_audio_lod_4')

  if slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, nch]
    with tf.variable_scope('upconv_4'):
      on_amount = lod - 4  # on at LOD 5
      _, audio_lod = up_block(output, audio_lod, on_amount, nch, kernel_len, 4, upsample=upsample)
      output = audio_lod # Audio LOD is main output for final layer

      # Summary for LOD level
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 5), 16000, max_outputs=10, family='G_audio_lod_5')

  elif slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      on_amount = lod - 4  # on at LOD 5
      output, audio_lod = up_block(output, audio_lod, on_amount, dim, kernel_len, 4, upsample=upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)

      # Summary for LOD level
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 5), 16000, max_outputs=10, family='G_audio_lod_5')

    # Layer 5
    # [16384, 64] -> [32768, nch]
    with tf.variable_scope('upconv_5'):
      on_amount = lod - 5  # on at LOD 6
      _, audio_lod = up_block(output, audio_lod, on_amount, nch, kernel_len, 2, upsample=upsample)
      output = audio_lod # Audio LOD is main output for final layer

      # Summary for LOD level
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('G_audio', audio_lod, 16000, max_outputs=10, family='G_audio_lod_6')

  elif slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      on_amount = lod - 4  # on at LOD 5
      output, audio_lod = up_block(output, audio_lod, on_amount, dim, kernel_len, 4, upsample=upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)

      # Summary for LOD level
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('G_audio', n_nn_upsamples(audio_lod, max_lod  - 5), 16000, max_outputs=10, family='G_audio_lod_5')

    # Layer 5
    # [16384, 64] -> [65536, nch]
    with tf.variable_scope('upconv_5'):
      on_amount = lod - 5  # on at LOD 6
      _, audio_lod = up_block(output, audio_lod, on_amount, nch, kernel_len, 4, upsample=upsample)
      output = audio_lod # Audio LOD is main output for final layer

      # Summary for LOD level
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('G_audio', audio_lod, 16000, max_outputs=10, family='G_audio_lod_6')

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
def PWaveGANDiscriminator(
    x,
    lod,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    labels=False,
    nlabels=1):
  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  if slice_len == 16384:
    # When operating on shorter audio (1 second), offset the LOD 
    # so that downconv_4 (the last layer for 1 second audio) is active
    # at LOD level 1
    lod += 1
    max_lod = 5
  else:
    max_lod = 6

  # Summary for LOD level
  if 'D_x/' in tf.get_default_graph().get_name_scope():
    tf.summary.audio('input_audio', x, 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod))

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    on_amount = lod - 5  # On at LOD 6, or 5 for 1 second audio
    output, audio_lod = down_block(output, x, on_amount, dim, kernel_len, 4)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Summary for LOD level
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 1), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 1))

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    on_amount = lod - 4  # On at LOD 5, or 4 for 1 second audio
    output, audio_lod = down_block(output, audio_lod, on_amount, dim * 2, kernel_len, 4)
    output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Summary for LOD level
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 2), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 2))

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    on_amount = lod - 3  # On at LOD 4, or 3 for 1 second audio
    output, audio_lod = down_block(output, audio_lod, on_amount, dim * 4, kernel_len, 4)
    output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Summary for LOD level
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 3), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 3))

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    on_amount = lod - 2  # On at LOD 3, or 2 for 1 second audio
    output, audio_lod = down_block(output, audio_lod, on_amount, dim * 8, kernel_len, 4)
    output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Summary for LOD level
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 4), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 4))

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    on_amount = lod - 1  # On at LOD 2, or 1 for 1 second audio
    output, audio_lod = down_block(output, audio_lod, on_amount, dim * 16, kernel_len, 4, last_block=(slice_len == 16384))
    output = batchnorm(output)
    output = lrelu(output)

    # Summary for LOD level
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.scalar('on_amount', on_amount)
      tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 5), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 5))

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      on_amount = lod  # On at LOD 1
      output, audio_lod = down_block(output, audio_lod, on_amount, dim * 32, kernel_len, 2, last_block=True)
      output = batchnorm(output)
      output = lrelu(output)

      # Summary for LOD level
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.scalar('on_amount', on_amount)
        tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 6, stride=2), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 6))

  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      on_amount = lod  # On at LOD 1
      output, audio_lod = down_block(output, audio_lod, on_amount, dim * 32, kernel_len, 4, last_block=True)
      output = batchnorm(output)
      output = lrelu(output)

      # Summary for LOD level
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.scalar('on_amount', on_amount)
        tf.summary.audio('input_audio', n_nn_upsamples(audio_lod, 6), 16000, max_outputs=10, family='D_audio_lod_{}'.format(max_lod - 6))

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
