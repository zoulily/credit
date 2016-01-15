# coding=utf-8

from abc import ABCMeta, abstractmethod

class Normalizer:
    __metaclass__ = ABCMeta

    """
    normalization function
    """
    def applyNormalization(self, data): pass