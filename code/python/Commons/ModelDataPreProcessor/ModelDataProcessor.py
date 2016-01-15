# coding=utf-8

from abc import ABCMeta, abstractmethod

class ModelDataProcessor:
    __metaclass__ = ABCMeta

    """
    give the final output that the model need
    """

    @abstractmethod
    def getModelNeededData(self): pass


    """
    get the pre-processed category data for model
    """
    @abstractmethod
    def getPreProcessedFlattenCategoryData(self, input, categoryInfo): pass

    """
    get the preprocessed numerical data for model
    """
    @abstractmethod
    def getPreProcessedFlattenNumericalData(self, input): pass

