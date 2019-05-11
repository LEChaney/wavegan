import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers import Dense, Conv2DTranspose, Embedding

def _l2normalize(v, eps=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + eps)

def power_iteration(W, u):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    v = _l2normalize(K.dot(u, K.transpose(W)))
    u = _l2normalize(K.dot(v, W))
    return u, v

class DenseSN(Dense):
    def build(self, input_shape):
        super(DenseSN, self).build(input_shape)

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        # Normalize it
        W_bar = W_reshaped / sigma
        # Reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)
        # Update weight tensor
        self.kernel = W_bar

        return super(DenseSN, self).call(inputs)
        
class ConvSN(Conv):
    def build(self, input_shape):
        super(ConvSN, self).build(input_shape)

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        # Normalize it
        W_bar = W_reshaped / sigma
        # Reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        # Update weight tensor
        self.kernel = W_bar

        return super(ConvSN, self).call(inputs)

class ConvSN1D(ConvSN):
    def __init__(self, *args, **kwargs):
        super(ConvSN1D, self).__init__(1, *args, **kwargs)

    def call(self, inputs, training=None):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(ConvSN1D, self).call(inputs, training=training)

class ConvSN2D(ConvSN):
    def __init__(self, *args, **kwargs):
        super(ConvSN2D, self).__init__(2, *args, **kwargs)

class ConvSN3D(ConvSN):
    def __init__(self, *args, **kwargs):
        super(ConvSN3D, self).__init__(3, *args, **kwargs)

class EmbeddingSN(Embedding):
    def build(self, input_shape):
        super(EmbeddingSN, self).build(input_shape)
        
        self.u = self.add_weight(shape=tuple([1, self.embeddings.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        W_shape = self.embeddings.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.embeddings, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        # Normalize it
        W_bar = W_reshaped / sigma
        # Teshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        # Update weight tensor
        self.embeddings = W_bar
            
        return super(EmbeddingSN, self).call(inputs)

class ConvSN2DTranspose(Conv2DTranspose):

    def build(self, input_shape):
        super(ConvSN2DTranspose, self).build(input_shape)

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        # Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        # Normalize it
        W_bar = W_reshaped / sigma
        # Reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        # Update weight tensor
        self.kernel = W_bar

        return super(ConvSN2DTranspose, self).call(inputs)
