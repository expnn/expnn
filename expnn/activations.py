# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano.tensor as T


def softmax(x):
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def time_distributed_softmax(x):
    import warnings
    warnings.warn("time_distributed_softmax is deprecated. Just use softmax!", DeprecationWarning)
    return softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return (x + abs(x)) / 2.0


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    """
    :param x: Input tensor variable that usually is the output of a bottom layer.
    :type x: theano.tensor.TensorVariable | theano.tensor.sharedvar.TensorSharedVariable
    :return: It returns the element-wise hard sigmoid of `x`, see the documentation of `theano hard sigmoid`_
      .. _`theano hard sigmoid`:
        http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.hard_sigmoid
    """
    return T.nnet.hard_sigmoid(x)


def linear(x):
    """
    :param x: Input tensor variable that could be the input of the network or the output of a bottom layer.
    :type x: theano.tensor.TensorVariable | theano.tensor.sharedvar.TensorSharedVariable
    :return: This function returns the variable that is passed in without any transformation.
    """
    return x


def exponential(x):
    """
    :param x: Input tensor variable that could be the input of the network or the output of a bottom layer.
    :type x: theano.tensor.TensorVariable | theano.tensor.sharedvar.TensorSharedVariable
    :return: The function returns the element-wise exponential of the input `x`.
    """
    return T.exp(x)


def normalization(x):
    y = T.sum(x, axis=-1, keepdims=True)
    return x / y

from .utils.generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
