from __future__ import absolute_import
# noinspection PyPep8Naming
import theano.tensor as T
from .utils.generic_utils import get_from_module


class Regularizer(object):
    def __init__(self):
        self.p = None
        self.layer = None

    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class WeightRegularizer(Regularizer):
    # noinspection PyShadowingNames
    def __init__(self, l1=0., l2=0.):
        super(WeightRegularizer, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.p = None

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += T.sum(abs(self.p)) * self.l1
        loss += T.sum(self.p ** 2) * self.l2
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2,
                "regularized parameter": self.p.name}


class ActivityRegularizer(Regularizer):
    # noinspection PyShadowingNames
    def __init__(self, l1=0., l2=0.):
        super(ActivityRegularizer, self).__init__()
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        loss += self.l1 * T.sum(T.mean(abs(self.layer.get_output(True)), axis=0))
        loss += self.l2 * T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


# noinspection PyShadowingNames
def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


# noinspection PyShadowingNames
def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)

identity = Regularizer


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer', instantiate=True, kwargs=kwargs)
