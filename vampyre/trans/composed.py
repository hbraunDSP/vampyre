#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComposedLT: Linear transform consisting of a composition of 2 or more
linear transforms
"""

from vampyre.trans.base import BaseLinTrans


class ComposedLT(BaseLinTrans):
    """
    Linear transform class implementing composition of two or more linear 
    transforms.
    
    :param transforms:  List of linear transform objects to be composed.
        Transforms are specified in order of application (i.e. innermost
        transform first). Note that this is opposite of the order in 
        which they would appear when left-multiplying a vector.
    """
    def __init__(self, transforms, name=None):
        shape0 = transforms[0].shape0
        shape1 = transforms[-1].shape1
        dtype0 = transforms[0].dtype0
        dtype1 = transforms[-1].dtype1
        super().__init__(shape0, shape1, dtype0, dtype1, 
            svd_avail=False,name=name)
        self.composed_transforms = transforms

    def dot(self, x):
        y = x
        for A in self.composed_transforms:
            y = A.dot(y)
        return y

    def dotH(self, y):
        x = y
        for A in reversed(self.composed_transforms):
            x = A.dotH(x)
        return x

    #@property
    #def svd_avail(self):
    #    #TODO implement the special case where all transforms are unitary
    #    return False
        
