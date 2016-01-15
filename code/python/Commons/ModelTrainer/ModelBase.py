# coding=utf-8

from abc import ABCMeta, abstractmethod

class ModelBase:
    __metaclass__ = ABCMeta
    def __init__(self,
                 flattenHeader=None,
                 flattenData=None,
                 flattenCategoryHeader=None,
                 flattenCategoryData=None,
                 flattenNumericalHeader=None,
                 flattenNumericalData=None,
                 flattenTargetHeader=None,
                 flattenTargetData=None):
        self.flattenHeader = flattenHeader
        self.flattenData = flattenData
        self.flattenCategoryHeader = flattenCategoryHeader
        self.flattenCategoryData = flattenCategoryData
        self.flattenNumericalHeader = flattenNumericalHeader
        self.flattenNumericalData = flattenNumericalData
        self.flattenTargetHeader = flattenTargetHeader
        self.flattenTargetData = flattenTargetData

    """
    Model Training using until converge method
    """

    @abstractmethod
    def trainModel(self):
        pass

    """
    additional step to deal with the target
    """
    @abstractmethod
    def targetTransform(self):
        pass

    """
    save the model to file
    """
    @abstractmethod
    def saveModelToFile(self, path):
        pass

    """
    validate the model
    """
    @abstractmethod
    def validateModel(self):
        pass