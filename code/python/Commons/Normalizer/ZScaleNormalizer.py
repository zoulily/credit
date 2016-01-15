# coding=utf-8

from Normalizer import Normalizer


class ZScaleNormalizer(Normalizer):


    """
    initialize
    """
    def __init__(self, capFactor):
        self.capFactor = capFactor

    """
    returned object is the data after normalization
    data should be in numpy array format
    """
    def applyNormalization(self, numericalData):
        (nRow, nColumn) = numericalData.shape
        for col in range(nColumn):
            colData = numericalData[:, col]
            std = colData.std()
            mean = colData.mean()
            upperBound = mean + self.capFactor * std
            lowerBound = mean - self.capFactor * std
            for row in range(nRow):
                if numericalData[row, col] > upperBound:
                    numericalData[row, col] = lowerBound
                elif numericalData[row, col] < lowerBound:
                    numericalData[row, col] = lowerBound
                numericalData[row, col] = (numericalData[row, col] - mean) / std
        return numericalData

