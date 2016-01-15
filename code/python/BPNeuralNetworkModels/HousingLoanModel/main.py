import sys
sys.path.append("../../Commons/")
sys.path.append("../../Commons/DataProcessor")
sys.path.append("../../Commons/ModelTrainer")
sys.path.append("../../Commons/Utils")
import numpy as np
import pandas as pd
import Utility as util

from pymongo import MongoClient
from MongoDataProcessor import MongoDataProcessor
from sklearn import preprocessing
from BPModelDataProcessor import BPModelDataProcessor
from BPModelTrainer import BPModelTrainer

loanType = 'HousingLoan'
version = 1.0
mongoAddress = ['192.168.1.125','192.168.1.126','192.168.1.127']
dbSource = MongoClient(mongoAddress, 27017)
dataBase = dbSource.RANK
mongoDataProcessor = MongoDataProcessor('1018', address=mongoAddress)
bpModelDataPrcocessor = BPModelDataProcessor()

allTags = mongoDataProcessor.getMetaTagList()

# get source data from mongo
# flattenHeader, flattenData = mongoDataProcessor.getFlattenTagData(additionalTags=['creditScore'])

#the test way ,may be not used in the future
flattenData = pd.read_csv('./Data/housing_source.csv')
(row,column)=np.shape(flattenData)
# columnScore = pd.Series(np.random.randint(100,200,row))
# flattenData['creditScore']=columnScore
targetHeader = ['creditScore']
targetData = np.random.randint(120,200,row)

categoryFrame = pd.DataFrame()
numericalFrame = pd.DataFrame()
categoryHeader = []
numericalHeader = []
# distinct from  numerical and categorical data
for column in flattenData.columns:
    if(mongoDataProcessor.isCategoryTag(column)):
        categoryFrame[column] = flattenData[column]
        categoryHeader.append(column)
    else:
        numericalFrame[column] = flattenData[column]
        numericalHeader.append(column)

categoryFrame.to_csv('./Data/categoryData.csv')
numericalFrame.to_csv('./Data/numericalData.csv')

# to array
categoryData = categoryFrame.values
numericalData = numericalFrame.values


# replace strange data with default value and make sure all the numbers are positive
fixedCategoryData = util.getFixedData(categoryData)
fixedNumericalData = util.getFixedData(numericalData)

# convert to ndarray
npCategoryData = np.array(fixedCategoryData, dtype=np.int)
npNumericalData = np.array(fixedNumericalData, dtype=np.float)

np.savetxt('./Data/npCategoryData.csv',npCategoryData,header=','.join(categoryHeader).encode('utf-8'),delimiter=',', fmt='%d',comments='')
np.savetxt('./Data/npNumericalData.csv',npNumericalData,header=','.join(numericalHeader).encode('utf-8'),delimiter=',', fmt='%.4f',comments='')

#replace missing value(-9) with mean, just for numerical data
npNumericalData = util.replaceMissingValueWithMean(npNumericalData)
np.savetxt('./Data/replaceMissingNumericalData.csv',npNumericalData,header=','.join(numericalHeader).encode('utf-8'),delimiter=',', fmt='%.4f',comments='')

#delete the column with constant values(std)
categoryHeader, npCategoryData = util.deleteColumnWithConstantValue((categoryHeader, npCategoryData))
numericalHeader, npNumericalData = util.deleteColumnWithConstantValue((numericalHeader, npNumericalData))
np.savetxt('./Data/reducedCategoryData.csv',npCategoryData,header=','.join(categoryHeader).encode('utf-8'),delimiter=',', fmt='%d',comments='')
np.savetxt('./Data/reducedNumericalData.csv',npNumericalData,header=','.join(numericalHeader).encode('utf-8'),delimiter=',', fmt='%.4f',comments='')

# z-score format
zscoreNumericalData = preprocessing.scale(npNumericalData)
np.savetxt('./Data/zscoreNumericalData.csv',zscoreNumericalData,header=','.join(numericalHeader).encode('utf-8'),delimiter=',', fmt='%.4f',comments='')

# bitformat for categoryData
preCategoryHeader, preCategoryData = bpModelDataPrcocessor.getPreProcessedFlattenCategoryData((categoryHeader, npCategoryData), mongoDataProcessor.getCategoryInfo())
np.savetxt('./Data/preCategoryData.csv',preCategoryData,header=','.join(preCategoryHeader).encode('utf-8'),delimiter=',', fmt='%d',comments='')

# PCA+KMeans for numericalData
PCAResultMap, transResultMap = bpModelDataPrcocessor.getAssociatedMapFromPCA((numericalHeader, zscoreNumericalData))
kmLists = bpModelDataPrcocessor.getKMeansListByCalculation((numericalHeader, zscoreNumericalData), (PCAResultMap, transResultMap), path='./Figures/')
bpModelDataPrcocessor.saveKMeansListToFile('./KMeansModel/', numericalHeader)
preNumericalHeader, preNumericalData = bpModelDataPrcocessor.getPreProcessedFlattenNumericalData((numericalHeader, zscoreNumericalData), dropTags=[])
np.savetxt('./Data/preNumericalData.csv',preNumericalData,header=','.join(preNumericalHeader).encode('utf-8'),delimiter=',', fmt='%d',comments='')

# ANN
bPModelTrainer = BPModelTrainer(
    flattenCategoryHeader=preCategoryHeader,
    flattenCategoryData=preCategoryData,
    flattenNumericalHeader=preNumericalHeader,
    flattenNumericalData=preNumericalData,
    flattenTargetHeader=targetHeader,
    flattenTargetData=targetData
)
# transport targetData to discrete value
bPModelTrainer.targetTransform()

# ANN train
bPModelTrainer.trainModel()
# validate
bPModelTrainer.validateModel()
# save
bPModelTrainer.saveModelToFile('Data/NNModel.xml')
