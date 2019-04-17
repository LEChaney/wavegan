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


def to_audio(inputs, nch):
  '''
  Converts feature map into an audio clip.
  '''
  assert(nch == 1 or nch ==2)

  with tf.variable_scope('to_audio'):
    def transform_input():
      return tf.layers.conv1d(inputs, filters=nch, kernel_size=1, strides=1, padding='same')
    
    def output_input():
      return inputs

    in_feature_maps = inputs.shape[2]
    output = tf.cond(tf.not_equal(in_feature_maps, nch), transform_input, output_input)
    output.set_shape([inputs.shape[0], inputs.shape[1], nch])

    # TODO: Add normalization
    output = tf.layers.conv1d(inputs, filters=nch, kernel_size=1, strides=1, padding='same')
    output = tf.nn.tanh(output)
    return output


def from_audio(inputs, out_feature_maps):
  '''
  Converts an input audio clip into a feature maps.
  '''
  with tf.variable_scope('from_audio'):
    # def transform_input():
    #   output = tf.layers.conv1d(inputs, filters=out_feature_maps, kernel_size=1, strides=1, padding='same')
    #   # TODO: Add normalization
    #   output = lrelu(output)
    #   return output
    
    # def output_input():
    #   return inputs

    # in_nch = inputs.shape[2]
    # assert(in_nch == 1 or in_nch == 2)

    # output = tf.cond(tf.not_equal(in_nch, out_feature_maps), transform_input, output_input)
    # output.set_shape([inputs.shape[0], inputs.shape[1], out_feature_maps])

    # TODO: Add normalization
    output = tf.layers.conv1d(inputs, filters=out_feature_maps, kernel_size=1, strides=1, padding='same')
    output = lrelu(output)
    return output


def up_block(inputs, audio_lod, on_amount, filters, kernel_len=9, stride=4, upsample='zeros'):
  '''
  Up Block
  '''
  def conv_layers():
    # TODO: Add normalization
    code = inputs
    code = conv1d_transpose(code, filters * 2, kernel_len, stride // 2, upsample=upsample)
    code = lrelu(code)
    code = conv1d_transpose(code, filters, kernel_len, stride // 2, upsample=upsample)
    return code

  def skip():
    with tf.variable_scope('skip'):
      skip_connection_code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)
      return skip_connection_code, nn_upsample(audio_lod, stride)

  def transition():
    with tf.variable_scope('transition'):
      code = conv_layers()
    
      # Blend this LOD block in over time
      nch = audio_lod.shape[2]
      out_audio_lod = to_audio(code, nch)
      out_audio_lod = lerp_clip(nn_upsample(audio_lod, stride), out_audio_lod, on_amount)

      return code, out_audio_lod

  def fully_on():
    with tf.variable_scope('fully_on'):
      code = conv_layers()

      nch = audio_lod.shape[2]
      out_audio_lod = to_audio(code, nch)

      return code, out_audio_lod

  def next_layer_fully_on():
    with tf.variable_scope('next_layer_fully_on'):
      code = conv_layers()
      
      # When the next layer is fully on we don't need to calculate an input audio lod for it
      # anymore, since it will only be using the 'code' output of this layer to calculate its
      # audio and code outputs.
      out_audio_lod = tf.zeros([tf.shape(audio_lod)[0], tf.shape(audio_lod)[1] * stride, tf.shape(audio_lod)[2]], dtype=tf.float32)

      return code, out_audio_lod

  code, out_audio_lod = tf.cond(on_amount <= 0.0, skip,
                lambda: tf.cond(on_amount <  1.0, transition,
                lambda: tf.cond(on_amount <= 2.0001, fully_on, # Small epsilon to prevent outputting blank audio from final up block 
                        next_layer_fully_on)))

  code.set_shape([inputs.shape[0], inputs.shape[1] * stride, filters])
  out_audio_lod.set_shape([audio_lod.shape[0], audio_lod.shape[1] * stride, audio_lod.shape[2]])
  assert(audio_lod.shape[2] == 1 or audio_lod.shape[2] == 2)

  return code, out_audio_lod


def down_block(inputs, audio_lod, on_amount, filters, kernel_len=9, stride=4):
  '''
  Down Block
  '''
  out_filters = filters * 2
  def conv_layers():
    # Blend this LOD block in over time
    # TODO: Add normalization
    # TODO: Make phase shuffle adjustable
    code = inputs
    code = tf.layers.conv1d(code, filters, kernel_len, stride // 2, padding='SAME')
    code = lrelu(code)
    code = apply_phaseshuffle(code, 2)
    code = tf.layers.conv1d(code, out_filters, kernel_len, stride // 2, padding='SAME')
    return code

  def next_layer_fully_off():
    with tf.variable_scope('next_layer_fully_off'):
      # Since the next layer is fully off, it will only need the output audio lod
      # from this layer. There is no need to calculate any kind of output code here.
      code = tf.zeros([inputs.shape[0], inputs.shape[1] // stride, out_filters], dtype=tf.float32)
      out_audio_lod = avg_downsample(audio_lod, stride)
      return code, out_audio_lod

  def skip():
    with tf.variable_scope('skip'):
      out_audio_lod = avg_downsample(audio_lod, stride)
      return from_audio(out_audio_lod, out_filters), out_audio_lod

  def transition():
    with tf.variable_scope('transition'):
      out_audio_lod = avg_downsample(audio_lod, stride)
      code = conv_layers()
      code = lerp_clip(from_audio(out_audio_lod, out_filters), code, on_amount)

      return code, out_audio_lod

  def fully_on():
    with tf.variable_scope('fully_on'):
      # When this layer is fully on we don't need to calculate the downsampled audio lod and
      # convert it to a code to blend between anymore. We can simply output the calculated
      # output code from the this block inputs. The next downsample block is also garanteed 
      # to be fully on, so it will also only need code output from this block.
      out_audio_lod = tf.zeros([audio_lod.shape[0], audio_lod.shape[1] // stride, audio_lod.shape[2]], dtype=tf.float32)
      code = conv_layers() 
      return code, out_audio_lod
    
  code, out_audio_lod = tf.cond(on_amount <= -1.0, next_layer_fully_off,
                lambda: tf.cond(on_amount <=  0.0, skip,
                lambda: tf.cond(on_amount <   1.0, transition,
                        fully_on)))

  code.set_shape([inputs.shape[0], inputs.shape[1] // stride, out_filters])
  out_audio_lod.set_shape([audio_lod.shape[0], audio_lod.shape[1] // stride, audio_lod.shape[2]])
  assert(audio_lod.shape[2] == 1 or audio_lod.shape[2] == 2)

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
    train=False):
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
  # We are reducing the dimensions of this model to allow for a 2 stage convolution in 
  # the up / down blocks. This should help the model to create more complex
  # non-linear mappings for each LOD level.
  dim = int(dim // 2.5)

  # Audio Summary
  # TODO: Allow using custom sample rate (not just hard coded to 16KHz)
  audio_lod = to_audio(output, nch)
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
    phaseshuffle_rad=0):
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

  # We are reducing the dimensions of this model to allow for a 2 stage convolution in 
  # the up / down blocks. This should help the model to create more complex
  # non-linear mappings for each LOD level.
  dim = int(dim // 2.5)

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
    output, audio_lod = down_block(output, audio_lod, on_amount, dim * 16, kernel_len, 4)
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
      output, audio_lod = down_block(output, audio_lod, on_amount, dim * 32, kernel_len, 2)
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
      output, audio_lod = down_block(output, audio_lod, on_amount, dim * 32, kernel_len, 4)
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
