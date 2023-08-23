import tensorflow as tf
from tensorflow.keras.layers import Lambda, concatenate, Add, Conv2D
from tensorflow.keras import datasets, layers, regularizers

def to_complex(x, n):
    return tf.complex(
        tf.cast(x[..., :n], dtype=tf.float32),
        tf.cast(x[..., n:], dtype=tf.float32),
    )

def to_real(x):
    return tf.concat([
        tf.math.real(x),
        tf.math.imag(x),
    ], axis=-1)

def _concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def _complex_from_half(x, n, output_shape):
    return Lambda(lambda x: to_complex(x, n), output_shape=output_shape)(x)

def complex_to_channels(image):            #output is of size (number of images, height, width*2)
    """Convert data from complex to channels."""
    image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
    shape_out = tf.concat(
        [tf.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out  

class CConv2D(tf.keras.Model):

    def __init__(self, num_outputs, kernel_size, padding=None, activation=None, kernel_initializer=None, kernel_regularizer=None, use_bias=None):
      super(CConv2D, self).__init__()
      self.num_outputs_2 = num_outputs
      self.num_outputs = num_outputs // 2
      self.kernel_size = kernel_size
      self.padding = padding
      self.kernel_initializer = kernel_initializer
      self.kernel_regularizer = kernel_regularizer
      self.use_bias = use_bias
      self.conv_real = layers.Conv2D(self.num_outputs, self.kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, use_bias=use_bias) #for real part of kernel
      self.conv_imag = layers.Conv2D(self.num_outputs, self.kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, use_bias=use_bias) #for imag part of kernel

    def build(self, input_shape):
      self.built = True

    def call(self, input):
      in_channels = input.shape[-1] 

      in_real = input[:,:,:,:in_channels//2]
      in_imag = input[:,:,:,in_channels//2:]
    
      real_real = self.conv_real(in_real)
      real_imag = self.conv_imag(in_real)
      imag_real = self.conv_real(in_imag)
      imag_imag = self.conv_imag(in_imag)

      out_real = real_real-imag_imag
      out_imag = imag_real+real_imag

      complex_output = tf.complex(out_real, out_imag)

      channels_output = complex_to_channels(complex_output)

      self.out_h = channels_output.shape[1]
      self.out_w = channels_output.shape[2]
      
      return channels_output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      shape[1] = self.out_h
      shape[2] = self.out_w
      shape[-1] = self.num_outputs * 2
      return tf.TensorShape(shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
            'num_outputs': self.num_outputs_2,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer':self.kernel_regularizer,
            'use_bias':self.use_bias,
        })
      return config


def conv2d_complex(x, n_filters, n_convs, activation='relu', output_shape=None, res=False, last_kernel_size=3):
    x_real_imag = _concatenate_real_imag(x)
    n_complex = output_shape[-1]
    for j in range(n_convs):
        x_real_imag = CConv2D(
            n_filters,
            3,
            activation=activation,
            padding='same',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(1e-6),
            # bias_regularizer=regularizers.l2(1e-6),
        )(x_real_imag)
    x_real_imag = CConv2D(
        2 * n_complex,
        last_kernel_size,
        activation='relu',
        padding='same',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(1e-6),
        # bias_regularizer=regularizers.l2(1e-6),
    )(x_real_imag)
    x_real_imag = _complex_from_half(x_real_imag, n_complex, output_shape)
    if res:
        x_final = Add()([x, x_real_imag])
    else:
        x_final = x_real_imag
    return x_final
