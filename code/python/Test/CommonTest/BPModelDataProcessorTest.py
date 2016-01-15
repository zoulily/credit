# coding=utf-8

import sys
import numpy as np
sys.path.append("../../Commons/DataProcessor")
sys.path.append("../../Commons/ModelDataPreProcessor")
sys.path.append("../../Commons/Utils")
sys.path.append("../../Commons/Normalizer")

from BPModelDataProcessor import BPModelDataProcessor
from MongoDataProcessor import MongoDataProcessor
import Utility as util
from ZScaleNormalizer import ZScaleNormalizer

mongoDataProessorTest = MongoDataProcessor('1019', address='192.168.1.120')
bpModelDataPrcocessor = BPModelDataProcessor()

# get the raw data from MongoDB
flattenHeader, flattenData = mongoDataProessorTest.getFlattenTagData(additionalTags=['creditScore'])
categoryHeader, categoryData = mongoDataProessorTest.getFlattenCategoryData()
numericalHeader, numericalData = mongoDataProessorTest.getFlattenNumericalData()
targetData = mongoDataProessorTest.getTargetTagData(targetTags=['creditScore'])
targetHead = ['creditScore']
categoryInfo = mongoDataProessorTest.getCategoryInfo()
tagDataByTagGroups = mongoDataProessorTest.getTagDataByTagGroups()

# replace strange data with default value
fixedCategoryData = util.getFixedData(categoryData)
fixedNumericalData = util.getFixedData(numericalData)

# convert data to numpy format
npCategoryData = np.array((fixedCategoryData), dtype=np.int)
npNumericalData = np.array((fixedNumericalData), dtype=np.float)

# replace missing value(-9) with mean, just for numerical data
npNumericalData, meanStdMap = util.replaceMissingWithMean((numericalHeader, npNumericalData))

# delete the column with constant values
npReducedCategoryHeader, npReducedCategoryData = util.deleteColumnWithConstantValue((categoryHeader, npCategoryData))
npReducedNumericalHeader, npReducedNumericalData = util.deleteColumnWithConstantValue((numericalHeader, npNumericalData))

with open('Data/ReducedData.csv', 'w') as f:
    print >> f,  ','.join(npReducedNumericalHeader).encode('utf-8')
    np.savetxt(f, npReducedNumericalData, delimiter=',', fmt='%d')  # please note the fmt arg

# normalize the data
normalizer = ZScaleNormalizer(capFactor=4)
npReducedNormNumericalData = normalizer.applyNormalization(npReducedNumericalData)

with open('Data/NormalizedData.csv', 'w') as f:
    print >> f,  ','.join(npReducedNumericalHeader).encode('utf-8')
    np.savetxt(f, npReducedNormNumericalData, delimiter=',', fmt='%.4f')  # please note the fmt arg


input = (npReducedCategoryHeader, npReducedCategoryData)
header, data = bpModelDataPrcocessor.getPreProcessedFlattenCategoryData(input, categoryInfo)
with open('Data/PreProcessedFlattenCategpryData.csv', 'w') as f:
    print >> f,  ','.join(header).encode('utf-8')
    np.savetxt(f, data, delimiter=',', fmt='%d')  # please note the fmt arg

# replace missing value (-9) with mean value

sortedMap, trans = bpModelDataPrcocessor.getAssociatedMapFromPCA((npReducedNumericalHeader, npReducedNumericalData))
kmLists = bpModelDataPrcocessor.getKMeansListByCalculation((npReducedNumericalHeader, npReducedNumericalData), (sortedMap, trans), path='./Figures/')
header, data = bpModelDataPrcocessor.getPreProcessedFlattenNumericalData((npReducedNumericalHeader, npReducedNumericalData), dropTags=['livePlace', 'occupation', 'brandModel'])
with open('Data/PreProcessedFlattenNumericalData.csv', 'w') as f:
    print >> f,  ','.join(header).encode('utf-8')
    np.savetxt(f, data, delimiter=',', fmt='%d')  # please note the fmt arg


