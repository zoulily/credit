# coding=utf-8
import sys
sys.path.append("../../Commons/")
sys.path.append("../../Commons/DataProcessor")
sys.path.append("../../Commons/Normalizer")
sys.path.append("../../Commons/ModelDataPreProcessor")
sys.path.append("../../Commons/ModelTrainer")
sys.path.append("../../Commons/Utils")

from BPModelDataProcessor import BPModelDataProcessor
from MongoDataProcessor import MongoDataProcessor
from BPModelTrainer import BPModelTrainer
import Utility as util
import numpy as np
import time
from pymongo import MongoClient
from ZScaleNormalizer import ZScaleNormalizer
loanType = 'CarLoan'
version = 1.0
mongoAddress = ['192.168.1.125','192.168.1.126','192.168.1.127']
dbSource = MongoClient(mongoAddress, 27017)
dataBase = dbSource.RANK
mongoDataProcessor = MongoDataProcessor('1019', address=mongoAddress)
bpModelDataPrcocessor = BPModelDataProcessor()
logPath = './log.txt'
# Step1: get all tags list
allTags = mongoDataProcessor.getMetaTagList()
util.printToFile(','.join(allTags), logPath, 'w')


# Step2: get Flatten Header/Data
flattenHeader, flattenData = mongoDataProcessor.getFlattenTagData(additionalTags=['creditScore'])
with open('Data/FlattenData.csv', 'w') as f:
    f.write(','.join(flattenHeader)+'\n')
    for dataRow in flattenData:
        print >> f,  ','.join(dataRow)  # encode('utf-8')

# Step3: get Target Header/Data
targetHeader, targetData = mongoDataProcessor.getTargetTagData(targetTags=['creditScore'])
targetData = np.array(targetData)
with open('Data/TargetData.csv', 'w') as f:
    f.write(','.join(targetHeader)+'\n')
    for dataRow in targetData:
        f.write(','.join(dataRow)+'\n')

# Step4: get Flatten Category and Numerical Data, the data are all of raw type!
# no target included
categoryHeader, categoryData = mongoDataProcessor.getFlattenCategoryData()
numericalHeader, numericalData = mongoDataProcessor.getFlattenNumericalData()
with open('Data/FlattenDataCategory.csv', 'w') as f:
    f.write(','.join(categoryHeader)+'\n')
    for dataRow in categoryData:
        f.write(','.join(dataRow)+'\n')
with open('Data/FlattenDataNumerical.csv', 'w') as f:
    f.write(','.join(numericalHeader)+'\n')
    for dataRow in numericalData:
        f.write(','.join(dataRow)+'\n')

# Step5: replace strange data with default value and make sure all the numbers are positive
fixedCategoryData = util.getFixedData(categoryData)
fixedNumericalData = util.getFixedData(numericalData)

# Step6: convert data to numpy format
npCategoryData = np.array((fixedCategoryData), dtype=np.int)
npNumericalData = np.array((fixedNumericalData), dtype=np.float)
with open('Data/npArrayCategoryData.csv', 'w') as f:
    print >> f,  ','.join(categoryHeader).encode('utf-8')
    np.savetxt(f, npCategoryData, delimiter=',', fmt='%d')  # please note the fmt arg
with open('Data/npArrayNumericalData.csv', 'w') as f:
    print >> f,  ','.join(numericalHeader).encode('utf-8')
    np.savetxt(f, npNumericalData, delimiter=',', fmt='%.4f')  # please note the fmt arg

# Step6: re-calculate variables in Numerical data
# Step6-1: occupation year
index = numericalHeader.index('occupationYear')
occupationYearData = npNumericalData[:, index]
# mean = meanStdMap['occupationYear'][0]
for row in range(len(occupationYearData)):
    if occupationYearData[row] != -9:
        occupationYearData[row] = occupationYearData[row] / (1000*3600*24) # to day

# Step6-2: carBuyDate, convert to elapsed days
index = numericalHeader.index('carBuyDate')
carBuyDateData = npNumericalData[:, index]
t = time.time()
for row in range(len(carBuyDateData)):
    if carBuyDateData[row] != -9:
        carBuyDateData[row] = (t*1000 - carBuyDateData[row])/(1000*3600*24)
with open('Data/ConvertedNumericalData.csv', 'w') as f:
    print >> f,  ','.join(numericalHeader).encode('utf-8')
    np.savetxt(f, npNumericalData, delimiter=',', fmt='%d')  # please note the fmt arg


# Step7: replace missing value(-9) with mean, just for numerical data
npNumericalData, statMap = util.replaceMissingWithMean((numericalHeader, npNumericalData))
with open('Data/replaceMissingNumericalData.csv', 'w') as f:
    print >> f,  ','.join(numericalHeader).encode('utf-8')
    np.savetxt(f, npNumericalData, delimiter=',', fmt='%.4f')  # please note the fmt arg

with open('Data/StatForNumericalData.csv', 'w') as f:
    f.write('name,max,min,mean,std\n')
    for key in statMap.keys():
        stats = statMap[key]
        f.write(key + ','+ str(stats[0]) + ',' + str(stats[1]) + ',' + str(stats[2]) + ',' + str(stats[3]) + '\n')

# Step7-1:clean age outliers
mean = statMap['age'][2] # max,min,mean,std
index = numericalHeader.index('age')
age = npNumericalData[:, index]
for row in range(len(age)):
    if age[row] > 80:
        age[row] = mean

# Step8: delete the column with constant values
npCategoryHeader, npCategoryData = util.deleteColumnWithConstantValue((categoryHeader, npCategoryData))
npNumericalHeader, npNumericalData = util.deleteColumnWithConstantValue((numericalHeader, npNumericalData))
with open('Data/ReducedCategoryData.csv', 'w') as f:
    print >> f,  ','.join(npCategoryHeader).encode('utf-8')
    np.savetxt(f, npCategoryData, delimiter=',', fmt='%d')  # please note the fmt arg
with open('Data/ReducedNumericalData.csv', 'w') as f:
    print >> f,  ','.join(npNumericalHeader).encode('utf-8')
    np.savetxt(f, npNumericalData, delimiter=',', fmt='%d')  # please note the fmt arg

# Step10: normalize the data
#normalizer = ZScaleNormalizer(capFactor=4)
#npNumericalData = normalizer.applyNormalization(npNumericalData, meanStdMap)
capFactor = 4
(nRow, nColumn) = npNumericalData.shape
for col in range(nColumn):
    tagName = npNumericalHeader[col]
    max, min, mean, std = statMap[tagName]
    upperBound = mean + capFactor * std
    lowerBound = mean - capFactor * std
    for row in range(nRow):
        if npNumericalData[row, col] > upperBound:
            npNumericalData[row, col] = upperBound
        elif npNumericalData[row, col] < lowerBound:
            npNumericalData[row, col] = lowerBound
        npNumericalData[row, col] = (npNumericalData[row, col] - mean) / std
with open('Data/NormalizedData.csv', 'w') as f:
    print >> f,  ','.join(npNumericalHeader).encode('utf-8')
    np.savetxt(f, npNumericalData, delimiter=',', fmt='%.4f')  # please note the fmt arg


# Step10-1: get the mean and std for each tag and store to mongo db

numStats = dataBase.NumStats
numStats.remove({"loanType": loanType})
for item in statMap.items():
    tagName = item[0]
    max, min, mean, std = item[1]
    numStats.insert_one(
        {
            "tagName": tagName,
            "max": int(max),
            "min": int(min),
            "mean" : int(mean),
            "std": float(std),
            "loanType": loanType,
            "version": version
        }
    )


# Step11: get pre-processed category data
categoryInfo = mongoDataProcessor.getCategoryInfo()
input = (npCategoryHeader, npCategoryData)
preCategoryHeader, preCategoryData = bpModelDataPrcocessor.getPreProcessedFlattenCategoryData(input, categoryInfo)
with open('Data/PreProcessedFlattenCategpryData.csv', 'w') as f:
    print >> f,  ','.join(preCategoryHeader).encode('utf-8')
    np.savetxt(f, preCategoryData, delimiter=',', fmt='%d')  # please note the fmt arg


CategoryTagHeader = dataBase.CategoryTagHeader
CategoryTagHeader.remove({"loanType":loanType})
for index in range(len(preCategoryHeader)):
    CategoryTagHeader.insert_one(
    {
        "tagName": preCategoryHeader[index],
        "updateTime": int(time.time()),
        "version": version,
        "loanType": loanType,
        "index" : index
    })


# Step12: get pre-processed numerical data
sortedMap, trans = bpModelDataPrcocessor.getAssociatedMapFromPCA((npNumericalHeader, npNumericalData))
kmLists = bpModelDataPrcocessor.getKMeansListByCalculation((npNumericalHeader, npNumericalData), (sortedMap, trans), path='./Figures/')
bpModelDataPrcocessor.saveKMeansListToFile('./KMeansModel/', npNumericalHeader)
preNumericalHeader, preNumericalData = bpModelDataPrcocessor.getPreProcessedFlattenNumericalData((npNumericalHeader, npNumericalData), dropTags=['livePlace', 'occupation', 'brandModel'])
with open('Data/PreProcessedFlattenNumericalData.csv', 'w') as f:
    print >> f,  ','.join(preNumericalHeader).encode('utf-8')
    np.savetxt(f, preNumericalData, delimiter=',', fmt='%d')  # please note the fmt arg


NumericalTagHeader = dataBase.NumericalTagHeader
NumericalTagHeader.remove({"loanType":loanType})
for index in range(len(preNumericalHeader)):
    NumericalTagHeader.insert_one(
    {
        "tagName": preNumericalHeader[index],
        "updateTime": int(time.time()),
        "version": version,
        "loanType": loanType,
        "index" : index
    })


#Step12-1: save the sorted map information to mongodb


KMeansTagsGroups = dataBase.KMeansTagsGroups
KMeansTagsGroups.remove({"loanType":loanType})
for index in range(len(sortedMap)):
    key, value = sortedMap[index]
    (col1, col2) = key.strip().split('#')
    intcol1 = int(col1)
    intcol2 = int(col2)
    tagName1 = npNumericalHeader[intcol1]
    tagName2 = npNumericalHeader[intcol2]
    KMeansTagsGroups.insert_one(
    {
        "tag1": tagName1,
        "tag2": tagName2,
        "updateTime": int(time.time()),
        "version": version,
        "loanType": loanType
    })



# Step13: set the parameters for bp model trainer
# Read from file to save time
with open('Data/PreProcessedFlattenCategpryData.csv') as f:
    header = f.readline()
    preCategoryHeader = header.split(',')
preCategoryData = np.genfromtxt('Data/PreProcessedFlattenCategpryData.csv', delimiter=',', skip_header=1, dtype=np.int)

with open('Data/PreProcessedFlattenNumericalData.csv') as f:
    header = f.readline()
    preNumericalHeader = header.split(',')
preNumericalData = np.genfromtxt('Data/PreProcessedFlattenNumericalData.csv', delimiter=',', skip_header=1, dtype=np.int)

with open('Data/TargetData.csv') as f:
    header = f.readline()
    targetHeader = header.split(',')
targetData = np.genfromtxt('Data/TargetData.csv', delimiter=',', skip_header=1, dtype=np.int)


bPModelTrainer = BPModelTrainer(
    flattenCategoryHeader=preCategoryHeader,
    flattenCategoryData=preCategoryData,
    flattenNumericalHeader=preNumericalHeader,
    flattenNumericalData=preNumericalData,
    flattenTargetHeader=targetHeader,
    flattenTargetData=targetData
)

# Step14: transform the target data to needed one
bPModelTrainer.targetTransform()

#Step15: generate the result distribution accroding to the tags
distMap = bPModelTrainer.generateRiskDistributionByRiskTags()
with open('Data/ResultDistribution.txt', 'w') as f:
    outStr = ''
    for item in distMap.items():
        tagGroup, value = item
        outStr += tagGroup + '\n'
        for item in value.items():
            cluster, value = item
            outStr += '\t'+ cluster + ': '
            sum = 0
            for im in value.items():
                r, v = im
                sum += v
            for item in value.items():
                result, value = item
                percent = 0
                if sum > 0:
                    percent = value*100.00 / sum
                outStr += '%d: %.2f,\t' % (result, percent)
            outStr += str(sum)
            outStr += '\n'
    f.write(outStr)

"""
# Step16: get risk tags
riskTags = bPModelTrainer.getRiskItems()
print ','.join(riskTags)

# Step17: write risk tags to mongodb
dbSource = MongoClient('192.168.1.120', 27017)
dataBase = dbSource.RANK
riskItem = dataBase.RankRiskItemWholeSet
riskItem.insert_one(
    {
        "tags": riskTags,
        "updateTime": int(time.time()),
        "effective": True
    }
)
"""

# Step18: train the model
bPModelTrainer.trainModel()


# Step19: save the model to file
bPModelTrainer.saveModelToFile('Data/NNModel.xml')

# Step20: validate
bPModelTrainer.validateModel()

