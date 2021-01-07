'''
Created on Sep 19, 2019
author: landeros10
'''
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
import tensorflow as tf
from tensorflow.keras import layers as lyrs
from tensorflow.keras.initializers import Constant
from tensorflow.keras import activations


def concat(batchList, axis=0):
    """
    batchList : list of tensors that need to be concatenated
    axis : axis along which to concatenate tensors
    """
    return lyrs.Concatenate(axis=axis)(batchList)


def sigmoid(x):
    return activations.sigmoid(x)


def relu(x):
    return activations.relu(x)


def prelu(x):
    return lyrs.PReLU(shared_axes=[1, 2])(x)


def tanh(x):
    """ Return tanh activation on layer x """
    return activations.tanh(x)


def batch_norm(x):
    """" Batch normalization in channels_last configuration """
    return lyrs.BatchNormalization()(x)


def dropout(x, rate, name="", dtype=None):
    """" Activation dropout """
    return lyrs.Dropout(rate, dtype=dtype)(x)


def dropout_bn(x, rate, dtype=None):
    if rate <= 0.0:
        return batch_norm(x)
    else:
        return batch_norm(dropout(x, rate, dtype=dtype))


def gaussNoise(x, stddev):
    return lyrs.GaussianNoise(stddev, dtype=x.dtype)(x)


def conv2d(x, features, filter_size,
           dilation=1,
           padding='valid',
           input_shape=None,
           dtype=None):
    """ Produces a two-dimensional convolution along axes 1 and 2.
    Presumes data organized as [N, H, W, C], where N indicates batch size
    and C indicates input channels

    Params
    x : input tensor
    features : number of features the output should have
    filter_size:  integer or tuple/list of 2 integers, specifying filter size
    rate : frequency at which neurons are dropped in dropout layer
    input_shape : if not None, specifies input shape for convoltion layer
    """
    # Default conv2d: bias, glorot initialization, valid padding, stride 1
    with tf.name_scope("conv2d"):
        if input_shape is not None:
            conv_layer = lyrs.Conv2D(features,
                                     filter_size,
                                     padding=padding,
                                     dilation_rate=dilation,
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=Constant(0.0),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     data_format="channels_last",
                                     input_shape=input_shape,
                                     dtype=dtype)(x)
        else:
            conv_layer = lyrs.Conv2D(features, filter_size,
                                     padding=padding,
                                     dilation_rate=dilation,
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=Constant(0.0),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     data_format="channels_last",
                                     dtype=dtype)(x)
        return conv_layer


def deconv2d(x, features, pool_size):
    """ Produces a two-dimensional transpose convolution along axes 1 and 2.
    Presumes data organized as [N, H, W, C], where N indicates batch size
    and C indicates input channels

    Params
    x : input tensor
    features : number of features the output should have
    pool_size : size of pooling operation to undo (usually 2)
    """
    with tf.name_scope("deconv2d"):
        deconv_layer = lyrs.Conv2DTranspose(features, pool_size,
                                            strides=pool_size,
                                            kernel_initializer='glorot_normal',
                                            bias_initializer=Constant(0.0),
                                            kernel_regularizer=None,
                                            bias_regularizer=None,
                                            data_format="channels_last")
        return deconv_layer(x)


def max_pool(x, n):
    return lyrs.MaxPooling2D(pool_size=(n, n))(x)


def avg_pool(x):
    return lyrs.GlobalAveragePooling2D()(x)


def linear(x, f, dtype=None):
    return lyrs.Dense(f, activation=None, dtype=dtype,
                      use_bias=True, bias_initializer='zeros')(x)


def crop_concat(x1, x2, f):
    """ Crops x1 to the shape of x2. Takes number of channels in x1 indicated
    features parameter.
    """
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # offsets for the top left corner of the crop
        offx = (x1_shape[1] - x2_shape[1]) // 2
        offy = (x1_shape[2] - x2_shape[2]) // 2
        offsets = [0, offx, offy, 0]
        size = [-1, x2_shape[1], x2_shape[2], f]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def crop_sum(x1, x2):
    """ Crops x1 to the shape of x2. Reduces x1 to single feature by
    convolution. Sums reduced x1 to x2.
    """
    with tf.name_scope("crop_and_sum"):
        x1 = conv2d(x1, 1, 1, 0.0)
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # offsets for the top left corner of the crop
        offx = (x1_shape[1] - x2_shape[1]) // 2
        offy = (x1_shape[2] - x2_shape[2]) // 2
        offsets = [0, offx, offy, 0]
        size = [-1, x2_shape[1], x2_shape[2], 1]
        x1_crop = tf.slice(x1, offsets, size)
        return x1_crop + x2


def dapi_add(dapi, input, joinType, f):
    if joinType == "concat":
        input = crop_concat(dapi, input, f)
    elif joinType == "sum":
        input = crop_sum(dapi, input)
    return input


def dapi_process(dapi, dapiFeat, filter_size, rate):
    dapi = conv2d(dapi, dapiFeat, filter_size)
    dapi = dropout_bn(dapi, rate)
    dapi = conv2d(dapi, dapiFeat, filter_size)
    return dropout_bn(dapi, rate)


def convert_sparse(input, size):
    return tf.cast(tf.round(input * (size)), tf.int32)
