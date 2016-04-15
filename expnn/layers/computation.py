# -*- coding: utf-8 -*-
from __future__ import absolute_import
# noinspection PyUnresolvedReferences
from six.moves import zip
import theano
# noinspection PyPep8Naming
import theano.tensor as T
import numpy as np
import logging
from ..utils.theano_utils import shared_zeros, floatX
from ..core import Param
from .. import activations, initializations, regularizers, constraints
from .. import float_t
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from ..utils.generic_utils import standardize_io
from time import time


class LayerType(type):
    def __new__(mcs, name, bases, dict_):
        dict_['logger'] = logging.getLogger('expnn.layers.computation.%s' % name)
        dict_['ids'] = [-1]
        return type.__new__(mcs, name, bases, dict_)


class Layer(object):
    __metaclass__ = LayerType

    # noinspection PyUnresolvedReferences
    def __init__(self, name='', outputs=(), inputs=()):
        self.params = []
        self.inputs = standardize_io(inputs)

        self.input_vars = {}
        self.output_vars = {}
        idx = max(self.ids)+1
        self.ids.append(idx)
        self.internal_name = self.__class__.__name__ + '_%d' % idx
        self.name = name if name else self.internal_name
        outputs = standardize_io(outputs) if outputs else '%sOut' % self.name
        self.outputs = [out if out else '%sOut%d' % (self.name, i) for i, out in enumerate(outputs)]

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    def perform(self):
        self.output_vars[True] = self.input_vars[True]
        self.output_vars[False] = self.output_vars[False]

    def get_output(self, is_train=False):
        return self.output_vars[is_train]

    def get_input(self, is_train=False):
        return self.input_vars[is_train]

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            if p.shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." % (p.shape, w.shape))
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

    def get_config(self):
        return {"type": self.__class__.__name__,
                "name": self.name,
                "param": [p.get_config for p in self.params]}

    def count_params(self):
        return sum([np.prod(p.shape.eval()) for p in self.params])


class Dropout(Layer):
    """
        Hinton's dropout.
    """
    def __init__(self, p, name='', outputs=(), inputs=(), seed=time()):
        super(Dropout, self).__init__(name, outputs, inputs)
        self.p = p
        self.srng = RandomStreams(seed=seed)

    def perform(self):
        for is_train in [True, False]:
            ins = self.get_input(is_train)
            if self.p > 0.:
                retain_prob = 1. - self.p
                if is_train:
                    ins *= self.srng.binomial(ins.shape, p=retain_prob, dtype=float_t)
                else:
                    ins *= retain_prob
            self.output_vars[is_train] = [ins]

    def get_config(self):
        return {"type": self.__class__.__name__,
                "name": self.name,
                "p": self.p,
                "param": [p.get_config for p in self.params]}


class Dense(Layer):
    """
        The regular fully connected NN layer.
    """

    # noinspection PyPep8Naming
    def __init__(self, input_dim, output_dim, name='', outputs=(), inputs=(),
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None):
        super(Dense, self).__init__(name, outputs, inputs)
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros(self.output_dim)

        self.params = [Param(name=self.name+'_W', v=self.W,
                             regularizer=regularizers.get(W_regularizer),
                             constraint=constraints.get(W_constraint)),
                       Param(name=self.name+'_b', v=self.b,
                             regularizer=regularizers.get(b_regularizer),
                             constraint=constraints.get(b_constraint))]

        self.regularizers = []
        self.activity_regularizer = regularizers.get(activity_regularizer)
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if weights is not None:
            self.set_weights(weights)

    def perform(self):
        self.output_vars[False] = self.activation(T.dot(self.input_vars[False], self.W) + self.b)
        self.output_vars[True] = self.activation(T.dot(self.input_vars[True], self.W) + self.b)

    def get_output(self, is_train=False):
        return self.output_vars[is_train]

    def get_config(self):
        return {"name": self.name,
                "type": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__,
                "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "params": [p.get_config() for p in self.params]}


class SimpleRNN(Layer):
    """
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    """

    # noinspection PyPep8Naming
    def __init__(self, input_dim, output_dim, name='', outputs=(), inputs=(),
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 WUb_regularizers=(None, None, None), WUb_constraints=(None, None, None),
                 truncate_gradient=-1, shift_left=False):

        super(SimpleRNN, self).__init__(name, outputs, inputs)
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.shift_left = shift_left

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros(self.output_dim)
        self.h0 = shared_zeros(shape=(1, self.output_dim), name='h0')

        # self.params = [self.W, self.U, self.b]
        self.params = [Param(name=self.name+'_W', v=self.W,
                             regularizer=regularizers.get(WUb_regularizers[0]),
                             constraint=constraints.get(WUb_constraints[0])),
                       Param(name=self.name+'_U', v=self.U,
                             regularizer=regularizers.get(WUb_regularizers[1]),
                             constraint=constraints.get(WUb_constraints[1])),
                       Param(name=self.name+'_b', v=self.b,
                             regularizer=regularizers.get(WUb_regularizers[2]),
                             constraint=constraints.get(WUb_constraints[2])),
                       Param(name=self.name+'_h0', v=self.h0)]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, x_t, h_tm1, u):
        """
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        """
        return self.activation(x_t + T.dot(h_tm1, u))

    def perform(self):
        assert self.nb_input == 1, "The number of inputs for %s must be 1" % self.__class__.__name__
        for is_train in [True, False]:          # ns:= nb_samples; nt:=nb_timesteps; d:=input_dim
            ins = self.input_vars[is_train][0]  # shape: (ns, nt, d)
            ins = ins.dimshuffle((1, 0, 2))     # shape: (nt, ns, d), because `scan` iterates over the first dimension
            y = T.dot(ins, self.W) + self.b     # shape: (nt, ns, do), where do:=dimension of output.
            h0 = T.unbroadcast(T.repeat(self.h0, ins.shape[1], axis=0), 1)  # (ns, do)

            outputs, updates = theano.scan(
                self._step,            # this will be called with arguments (sequences[i], outputs[i-1], non_sequences)
                sequences=y,           # tensors to iterate over, inputs to `_step`. y[i] is the input at timestep `i`
                outputs_info=h0,       # initialization of the output. Input to `_step` with default tap=-1.
                non_sequences=self.U,  # static inputs to `_step`
                truncate_gradient=self.truncate_gradient, strict=True)
            assert len(updates) == 0, "Updates not empty!"

            if self.shift_left:
                res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
                self.output_vars[is_train] = [res[:-1]]
            else:
                self.output_vars[is_train] = [outputs]

    def get_config(self):
        return {"name": self.name,
                "type": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "activation": self.activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "shift_left": self.shift_left,
                "param": [p.get_config for p in self.params]}


class LangSimpleRNN(SimpleRNN):

    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super(LangSimpleRNN, self).__init__(input_dim, output_dim, *args, shift_left=True, **kwargs)


class LSTM(Layer):
    """
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    """

    # noinspection PyPep8Naming
    def __init__(self, input_dim, output_dim, name='', outputs=(), inputs=(),
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 input_activation='tanh', gate_activation='hard_sigmoid', output_activation='tanh',
                 weights=None, WRb_regularizers=(None, None, None), WRb_constraints=(None, None, None),
                 truncate_gradient=-1, shift_left=False):

        super(LSTM, self).__init__(name, outputs, inputs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.shift_left = shift_left

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.input_activation = activations.get(input_activation)
        self.gate_activation = activations.get(gate_activation)
        self.output_activation = activations.get(output_activation)

        device = initializations.TMP_CPU_DEVICE
        W_z = self.init((self.input_dim, self.output_dim), device=device).get_value(borrow=True)
        R_z = self.inner_init((self.output_dim, self.output_dim), device=device).get_value(borrow=True)

        W_i = self.init((self.input_dim, self.output_dim), device=device).get_value(borrow=True)
        R_i = self.inner_init((self.output_dim, self.output_dim), device=device).get_value(borrow=True)

        W_f = self.init((self.input_dim, self.output_dim), device=device).get_value(borrow=True)
        R_f = self.inner_init((self.output_dim, self.output_dim), device=device).get_value(borrow=True)

        W_o = self.init((self.input_dim, self.output_dim), device=device).get_value(borrow=True)
        R_o = self.inner_init((self.output_dim, self.output_dim), device=device).get_value(borrow=True)

        self.h0 = shared_zeros(shape=(1, self.output_dim), name='h0')
        self.c0 = shared_zeros(shape=(1, self.output_dim), name='c0')

        W = np.vstack((W_z[np.newaxis, :, :],
                       W_i[np.newaxis, :, :],
                       W_f[np.newaxis, :, :],
                       W_o[np.newaxis, :, :]))  # shape = (4, input_dim, output_dim)
        R = np.vstack((R_z[np.newaxis, :, :],
                       R_i[np.newaxis, :, :],
                       R_f[np.newaxis, :, :],
                       R_o[np.newaxis, :, :]))  # shape = (4, output_dim, output_dim)
        self.W = theano.shared(W, name='i2h', borrow=True)
        self.R = theano.shared(R, name='h2h', borrow=True)
        self.b = theano.shared(np.zeros(shape=(4, self.output_dim), dtype=theano.config.floatX),
                               name='bias', borrow=True)

        # self.params = [self.W, self.R, self.b]
        self.params = [Param(name=self.name+'_W', v=self.W,
                             regularizer=regularizers.get(WRb_regularizers[0]),
                             constraint=constraints.get(WRb_constraints[0])),
                       Param(name=self.name+'_R', v=self.R,
                             regularizer=regularizers.get(WRb_regularizers[1]),
                             constraint=constraints.get(WRb_constraints[1])),
                       Param(name=self.name+'_b', v=self.b,
                             regularizer=regularizers.get(WRb_regularizers[2]),
                             constraint=constraints.get(WRb_constraints[2])),
                       Param(name=self.name+'_h0', v=self.h0),
                       Param(name=self.name+'_c0', v=self.c0)]

        if weights is not None:
            self.set_weights(weights)

    # noinspection PyPep8Naming
    def _step(self,
              Y_t,            # sequence     (ns, 4, do)
              h_tm1, c_tm1,   # output_info  (ns, do)
              R):             # non_sequence (4, do, do)
        G_tm1 = T.dot(h_tm1, R)  # (ns, 4, do)
        M_t = Y_t + G_tm1        # (ns, 4, do)
        z_t = self.input_activation(M_t[:, 0, :])  # (ns, do)
        ifo_t = self.gate_activation(M_t[:, 1:, :])
        i_t = ifo_t[:, 0, :]     # (ns, do)
        f_t = ifo_t[:, 1, :]
        o_t = ifo_t[:, 2, :]
        c_t = f_t * c_tm1 + i_t * z_t
        h_t = o_t * self.output_activation(c_t)
        return h_t, c_t

    # noinspection PyPep8Naming
    def perform(self):
        assert self.nb_input == 1, "The number of inputs for %s must be 1" % self.__class__.__name__
        for is_train in [True, False]:      # ns:= nb_samples; nt:=nb_timesteps; d:=input_dim
            X = self.get_input(is_train)    # (ns, nt, d)
            X = X.dimshuffle((1, 0, 2))     # (nt, ns, d)
            Y = T.dot(X, self.W) + self.b   # (nt, ns, 4, do), where `do` is the output dimension.
            h0 = T.repeat(self.h0, X.shape[1], axis=0)  # (ns, do)
            c0 = T.repeat(self.c0, X.shape[1], axis=0)  # (ns, do)

            [outputs, _], updates = theano.scan(
                self._step,
                sequences=Y,
                outputs_info=[h0, c0],
                non_sequences=[self.R],
                truncate_gradient=self.truncate_gradient, strict=True)

            if self.shift_left:
                res = T.concatenate([h0.dimshuffle('x', 0, 1), outputs], axis=0).dimshuffle((1, 0, 2))
                self.output_vars[is_train] = [res[:-1]]
            else:
                self.output_vars[is_train] = [outputs]

    def get_config(self):
        return {"name": self.name,
                "type": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "inner_init": self.inner_init.__name__,
                "forget_bias_init": self.forget_bias_init.__name__,
                "input_activation": self.input_activation.__name__,
                "gate_activation": self.gate_activation.__name__,
                "truncate_gradient": self.truncate_gradient,
                "shift_left": self.shift_left,
                "param": [p.get_config for p in self.params]}


class LangLSTM(LSTM):

    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super(LangLSTM, self).__init__(input_dim, output_dim, *args, shift_left=True, **kwargs)


class Filter(Layer):
    def __init__(self, name='', outputs=(), inputs=()):
        super(Filter, self).__init__(name, outputs, inputs)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def perform(self):
        mask = self.mask
        for is_train in [True, False]:
            y = self.get_input(is_train)
            y *= mask
            self.output_vars[is_train] = [y]

    def get_config(self):
        config = super(Filter, self).get_config()
        config['mask'] = {"shape": self.mask.shape, "nonzero": self.mask.sum()} if self.mask else None
        return config


class LastTimeStep(Filter):
    def __init__(self, name='', outputs=(), inputs=()):
        super(LastTimeStep, self).__init__(name, outputs, inputs)

    def perform(self):
        mask = self.mask
        assert mask.ndim == 2, 'Only 2D mask are supported'
        ind = T.switch(T.eq(mask[:, -1], 1.), mask.shape[-1], T.argmin(mask, axis=-1)).astype('int32')
        for is_train in [True, False]:
            y = self.get_input(is_train)
            res = y[T.arange(mask.shape[0], dtype='int32'), ind]
            self.output_vars[is_train] = [res]


