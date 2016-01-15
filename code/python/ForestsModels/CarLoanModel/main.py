
import sys
sys.path.append("../../Commons/")
sys.path.append("../../Commons/DataProcessor")
sys.path.append("../../Commons/ModelTrainer")
sys.path.append("../../Commons/Utils")
import numpy as np
import Utility as util

from pymongo import MongoClient
from MongoDataProcessor import MongoDataProcessor
from sklearn import preprocessing
from BPModelDataProcessor import BPModelDataProcessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation


def VarScore(estimator,x,y):
    sy = estimator.predict(x)
    return np.sum((sy-y)**2)

loanType = 'CarLoan'
version = 1.0
dbSource = MongoClient('192.168.1.120', 27017)
dataBase = dbSource.RANK
mongoDataProcessor = MongoDataProcessor('1019', address='192.168.1.120')
bpModelDataPrcocessor = BPModelDataProcessor()
#get flatten data
flattenHeader, flattenData = mongoDataProcessor.getFlattenTagData(additionalTags=['creditScore'])
np.savetxt('./Data/FlattenData.csv',flattenData,header=','.join(flattenHeader).encode('utf-8'),delimiter=',', fmt='%s',comments='')
# get target data
targetHeader, targetData = mongoDataProcessor.getTargetTagData(targetTags=['creditScore'])
# trans targetData to number
targetData = np.array(targetData)
targetData = np.array(targetData.flatten(),dtype = np.int32)
np.savetxt('./Data/TargetData.csv',targetData,header=','.join(targetHeader).encode('utf-8'),delimiter=',',fmt='%s',comments='')
# get source data from mongo
categoryHeader, categoryData = mongoDataProcessor.getFlattenCategoryData()
numericalHeader, numericalData = mongoDataProcessor.getFlattenNumericalData()
np.savetxt('./Data/FlattenDataCategory.csv',categoryData,header=','.join(categoryHeader).encode('utf-8'),delimiter=',',fmt='%s',comments='')
np.savetxt('./Data/FlattenDataNumerical.csv',numericalData,header=','.join(numericalHeader).encode('utf-8'),delimiter=',',fmt='%s',comments='')



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
header = categoryHeader + numericalHeader
data = np.c_[npCategoryData,zscoreNumericalData]

np.savetxt('./Data/data.csv',data,header=','.join(header).encode('utf-8'),delimiter=',', fmt='%.4f',comments='')

rfr = ExtraTreesRegressor()
# rfr.fit(data[500:,:],targetData[500:])
#
# np.savetxt('./Data/1.csv',rfr.predict(data),delimiter=',', fmt='%.4f',comments='')
# np.savetxt('./Data/2.csv',targetData,delimiter=',', fmt='%.4f',comments='')
scores = cross_validation.cross_val_score(rfr,data,targetData,scoring=VarScore,cv=4);
print(scores)
print("accuracy:%0.2f (+/- %0.2f)" %(scores.mean(),scores.std()))

