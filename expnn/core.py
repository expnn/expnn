#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .constraints import Constraint
from .regularizers import Regularizer


class SubTensorInfo(object):
    def __init__(self, subset, base, index, shape=None):
        super(SubTensorInfo, self).__init__()
        self.variable = subset
        self.base = base
        self.idx = index
        if shape is not None:
            self.shape = shape
        else:
            self.shape = tuple(base.shape.eval())


class Param(object):
    def __init__(self, name, v, v_info=None,
                 regularizer=Regularizer(), constraint=Constraint(), updates=()):
        """
        :param v: Underlying (shared) tensor variable to store the parameters
        :param name: The name of this parameter
        :param v_info: the information of the variable `v`
        :type v_info: SubTensorInfo
        :param regularizer: the regularizer for this parameter
        :param constraint: the constraints for this parameter
        :param updates: updates for this parameter.
        :return: None
        """
        self.v = v
        self.v_shared = True
        if v_info is not None:
            assert id(v_info.variable) == id(v), "subtensor and its info does not match"
            self.shape = v_info.shape
            self.base = v_info.base
            self.idx = v_info.idx
            self.v_shared = False
        else:
            self.shape = v.shape.eval()
            self.base = v
            self.idx = None

        self.name = name
        self.v.name = name
        self.regularizer = regularizer
        self.constraint = constraint
        self.updates = list(updates)
        self.regularizer.set_param(self.v)

    def __getattr__(self, attr):
        return getattr(self.v, attr) if self.v_shared else getattr(self.base, attr)

    def set_name(self, name):
        self.v.name = name
        self.name = name

    def get_config(self):
        return {"name": self.name,
                "regularizer": self.regularizer.get_config(),
                "constraint": self.constraint.get_config(),
                "updates": self.updates,
                "param": {"shape": self.shape, 'type': self.v.type}}

