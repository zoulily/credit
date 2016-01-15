# coding=utf-8

'''
using the built model for calculation
'''

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
from pymongo import MongoClient
from ZScaleNormalizer import ZScaleNormalizer
from sklearn.externals import joblib
from pybrain.tools.customxml.networkreader import NetworkReader
import time

def retuneScoreForPersonalConsumeLoanModelFake(header, data, rowNum):
    offset = 0
    checkList = [
        'personalIncome',
        'familyIncome',
        'houseTotalAmount',
        'carTotalAmount',
        'creditSingleHighestAmount',
        'creditOverdueNinetyNum',
        'houseLoanOverdueLongestTime',
        'otherLoanBalance',
        'due',
        'financeAmount',
        'guaranteeWay',
        'extraGuarantee'
    ]
    checkMap = {}
    for item in checkList:
        checkMap[item] = -9 # default number
    dataRow = data[rowNum]
    for key in checkMap.keys():
        try:
            index = header.index(key)
            itemData = dataRow[index]
            if int(itemData) >= 0:
                checkMap[key] = int(itemData)
        except:
            print >> sys.stdout, key + 'not in tags or value is empty'
    for key in checkMap.keys():
        if key == 'personalIncome' and checkMap[key] >= 0:
            offset += checkMap[key] / 100
        elif key == 'familyIncome' and checkMap[key] >= 0:
            if checkMap['personalIncome'] >= 0:
                offset += (checkMap[key] - checkMap['personalIncome']) / 200
        elif key == 'houseTotalAmount' and checkMap[key] >= 0:
            netAmount = checkMap[key]
            if checkMap['carTotalAmount'] >= 0:
                netAmount += checkMap['carTotalAmount']
            if checkMap['otherLoanBalance'] >= 0:
                netAmount -= checkMap['otherLoanBalance']
            offset += netAmount / 5000
        elif key == 'creditSingleHighestAmount' and checkMap[key] >= 0:
            offset += checkMap[key] / 1000
        elif key == 'creditOverdueNinetyNum' and checkMap[key] >= 0:
            offset -= checkMap[key] * 50
        elif key == 'houseLoanOverdueLongestTime' and checkMap[key] >= 0:
            offset -= checkMap[key] / 30 * 30
        elif key == 'due' and checkMap[key] >= 0:
            if checkMap['financeAmount'] > 0:
                if checkMap['financeAmount'] / (checkMap[key]/30+1) < checkMap['familyIncome'] / 2:
                    offset += 100
                else:
                    offset -= 100
        elif key == 'guaranteeWay' and checkMap[key] >= 0:
            offset += 100
        elif key == 'extraGuarantee' and checkMap[key] >= 0:
            offset += 50
    return offset

logPath = '../../logs/log.txt'
loanType = 'CarLoan'
missingValue = '-9'

dbAddressIP = ['192.168.1.125','192.168.1.126','192.168.1.127'] #local
# dbAddressIP = '10.251.255.163' # aliyun

dbSource = MongoClient(dbAddressIP, 27017)
dataBase = dbSource.RANK
bpModelDataPrcocessor = BPModelDataProcessor()

# Step1: Get data from database
# tmp_xxx includes the information
# 1001: personal loan
# 1019: car loan
# 1018: house loan
mongoDataProcessor = MongoDataProcessor(['1001'], address=dbAddressIP)
header, data = mongoDataProcessor.getFlattenTagData(computeNeeded=True, fromTraining=False) # get only those records which need computation
data = util.getFixedData(data)
projectIds = mongoDataProcessor.getProjectId()
appIds = mongoDataProcessor.getAppId()
itemTypes = mongoDataProcessor.getItemType()
modelComputed = mongoDataProcessor.getModelComputed()
riskItemComputed = mongoDataProcessor.getRiskItemComputed()
if len(data) == 0 or len(projectIds) == 0 or appIds == 0:
    util.printToFile(str(time.time()) + ': no increamental data attached!', logPath)
    sys.exit()

# Step2: Reassemble headers
flattenCategoryHeaderTmp = {}
flattenNumericalHeaderTmp = {}
flattenCategoryHeader = []
flattenNumericalHeader = []
categoryHeader = []
numericalHeader = []
dbCategoryTagHeader = dataBase.CategoryTagHeader.find({"loanType": loanType}, {'itemType':1, 'tagName':1, 'index':1,'_id':0})
dbNumericalTagHeader = dataBase.NumericalTagHeader.find({"loanType": loanType},{'itemType':1, 'tagName':1, 'index':1,'_id':0})
for item in dbCategoryTagHeader:
    flattenCategoryHeaderTmp[item['index']] = item['tagName']
for item in dbNumericalTagHeader:
    flattenNumericalHeaderTmp[item['index']] = item['tagName']
for key in sorted(flattenCategoryHeaderTmp.keys()):
    flattenCategoryHeader.append(flattenCategoryHeaderTmp[key])
for key in sorted(flattenNumericalHeaderTmp.keys()):
    flattenNumericalHeader.append(flattenNumericalHeaderTmp[key])
index = 0
while index < len(flattenCategoryHeader):
    name, flag = flattenCategoryHeader[index].split('_')
    missingName = name + '_-9'
    indexAfter = flattenCategoryHeader.index(missingName)
    categoryHeader.append(name)
    index = indexAfter + 1
index = 0
while index < len(flattenNumericalHeader):
    tagName1, remain = flattenNumericalHeader[index].split('_and_')
    tagName2 = remain.split('_', 1)[0]
    if tagName1 not in numericalHeader:
        numericalHeader.append(tagName1)
    if tagName2 not in numericalHeader:
        numericalHeader.append(tagName2)
    index += 1
print flattenCategoryHeader
print flattenNumericalHeader
print categoryHeader
print numericalHeader

# Step3: get flatten category data
categoryTagIndex = []
categoryDataTmp = []
flattenCategoryData = []
for tagName in categoryHeader:
    if tagName in header:
        categoryTagIndex.append(header.index(tagName))
    else:
        categoryTagIndex.append(-1)

for row in data:
    rowData = []
    for index in categoryTagIndex:
        if index >= 0:
            rowData.append(row[index])
        else:
            rowData.append(missingValue)
    categoryDataTmp.append(rowData)

categoryData = np.array(categoryDataTmp, dtype=np.int)
categoryInfo = mongoDataProcessor.getCategoryInfo()
flattenCategoryData = bpModelDataPrcocessor.getPreProcessedFlattenCategoryData((categoryHeader, categoryData), categoryInfo)[1]
with open('Data/ResultFlattenCategoryData.csv', 'w') as f:
    f.write( ','.join(flattenCategoryHeader)+'\n')
    np.savetxt(f, flattenCategoryData, delimiter=',', fmt='%d')

# Step4: get flatten numerical data
numStats = dataBase.NumStats
meanStdMap = {}
mscollection = numStats.find({})
for item in mscollection:
    std = item['std']
    mean = item['mean']
    tagName = item['tagName']
    meanStdMap[tagName] = (mean, std)
numericalTagIndex = []
for tagName in numericalHeader:
    if tagName in header:
        numericalTagIndex.append(header.index(tagName))
    else:
        numericalTagIndex.append(-1)
#variable trasform

# special treatment for occupationYear & carBuyDate
numericalDataTmp = []
for row in data:
    rowData = []
    for index in numericalTagIndex:
        if index < 0:
            rowData.append(missingValue)
            continue
        tagName = header[index]
        value = 0
        try:
            value = float(row[index])
            if value == -9 or value < 0:  # missing value or negtive value
                value = meanStdMap[tagName][0]
            elif tagName == 'occupationYear':
                value = value / (1000*3600*24)
            elif tagName == 'carBuyDate':
                value = (time.time()*1000 - float(row[index])) / (1000*3600*24)
        except:
            value = meanStdMap[tagName][0]
        rowData.append(value)
    numericalDataTmp.append(rowData)
numericalData = np.array(numericalDataTmp, dtype=np.float)

# normalize the data
capFactor = 4
(nRow, nColumn) = numericalData.shape
for col in range(nColumn):
    tagName = numericalHeader[col]
    mean, std = meanStdMap[tagName]
    upperBound = mean + capFactor * std
    lowerBound = mean - capFactor * std
    for row in range(nRow):
        if numericalData[row, col] > upperBound:
            numericalData[row, col] = upperBound
        elif numericalData[row, col] < lowerBound:
            numericalData[row, col] = lowerBound
        numericalData[row, col] = (numericalData[row, col] - mean) / std
with open('Data/ResultNormalizedData.csv', 'w') as f:
    print >> f,  ','.join(numericalHeader).encode('utf-8')
    np.savetxt(f, numericalData, delimiter=',', fmt='%.4f')  # please note the fmt arg

#get kmeans clustering in sequence
index = 0
flattenNumericalData = None
while index < len(flattenNumericalHeader):
    tagNameCombined = flattenNumericalHeader[index]
    clusterName = tagNameCombined[0:-2]
    indexAfter = index + 1
    while indexAfter < len(flattenNumericalHeader) and flattenNumericalHeader[indexAfter].find(clusterName) >= 0:
        indexAfter += 1
    tagName1 = tagNameCombined.split('_')[0]
    tagName2 = tagNameCombined.split('_')[2]
    kmeansFileName = tagName1 + '_' + tagName2 + '.pkl'
    dataIndex1 = numericalHeader.index(tagName1)
    dataIndex2 = numericalHeader.index(tagName2)
    colData1 = numericalData[:, dataIndex1]
    colData2 = numericalData[:, dataIndex2]
    kmAlg = joblib.load('KMeansModel/'+kmeansFileName)
    predict = kmAlg.predict(np.c_[colData1, colData2])
    offset = indexAfter - index
    rowNumber = len(colData1)
    vector = np.zeros((rowNumber, kmAlg.n_clusters), dtype=np.int32)
    for rn in range(rowNumber):
        singlePre = predict[rn]
        vector[rn][singlePre] = 1
    if flattenNumericalData is None:
        flattenNumericalData = vector
    else:
        flattenNumericalData = np.c_[flattenNumericalData, vector]
    index = indexAfter

#flattenNumericalData = bpModelDataPrcocessor.getPreProcessedFlattenNumericalData((categoryHeader, categoryData), categoryInfo)[1]
with open('Data/ResultNumericalData.csv', 'w') as f:
    f.write(','.join(flattenNumericalHeader)+'\n')
    np.savetxt(f, flattenNumericalData, delimiter=',', fmt='%d')

# get the model and activate, write result back with app id and item id
modelPath = 'Data/NNModel.xml'
net = NetworkReader.readFrom(modelPath)
attributeSet = np.c_[flattenNumericalData, flattenCategoryData]

def getAlignedScore(result, coefficient=[0,150,160,180,300]):
    score = 0.0
    if len(result)+1 != len(coefficient):
        print ('length not equal when calculating score')
    else:
            #for e in range(len(result)):
            #    score += result[e]*coefficient[e]
        cat =  np.argmax(result)
        score = coefficient[cat] + result[cat] * (coefficient[cat+1] - coefficient[cat])
    return score

scoreList = []
ratingList = []
coefficient = 3.5
for rowNumber in range(len(attributeSet)):
    rowData = attributeSet[rowNumber]
    tar = net.activate(rowData)
    score = getAlignedScore(tar) * coefficient
    score += retuneScoreForPersonalConsumeLoanModelFake(header, data, rowNumber)
    if score > 950:
        score = 950
    if score < 50:
        score = 50
    if score < 350:
        rating = 'D'
    elif score < 500:
        rating = 'C'
    elif score < 537.5:
        rating = 'C+'
    elif score < 575:
        rating = 'C++'
    elif score < 612.5:
        rating = 'B-'
    elif score < 650:
        rating = 'B'
    elif score < 687.5:
        rating = 'B+'
    elif score < 725:
        rating = 'B++'
    elif score < 762.5:
        rating = 'A-'
    elif score < 800:
        rating = 'A'
    elif score < 900:
        rating = 'A+'
    else:
        rating = 'A++'
    scoreList.append(score)
    ratingList.append(rating)
if len(projectIds) != len(appIds) or len(projectIds) != len(scoreList):
    raise Exception('project length doesn\'t equal to the length of score List!')
now = time.time()
rankCredit = dataBase.RankCredit
rankCreditHis = dataBase.RankCreditHis
for rowNumber in range(len(projectIds)):
    try:
        itemType = itemTypes[rowNumber]
        score = scoreList[rowNumber]
        rating = ratingList[rowNumber]
        riskItemCmpt = riskItemComputed[rowNumber]
        modelCmpt = modelComputed[rowNumber]
        # finds = rankCredit.find({"itemId" : projectIds[rowNumber]})
        # if finds.count() > 0: # if the record is already existing and the user doesn't require re-estimate while he modified the tag, then show the last score to him.
        #    for element in finds:
        #        if modelCmpt is True:
        #            score = element['score']
        #            rating = element['rating']
        rankCredit.update_one(
            {"itemId" : projectIds[rowNumber]},
            {"$set":{
                "appid" : appIds[rowNumber],
                "itemId" : projectIds[rowNumber],
                "itemType": itemType,
                "score" : score,
                "rating": rating,
                "updateTime": now,
                "updateDate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "effective": True}
             },
             upsert=True
        )

        rankCreditHis.insert_one(
            {
                "appid" : appIds[rowNumber],
                "itemId" : projectIds[rowNumber],
                "itemType": itemType,
                "score" : score,
                "rating": rating,
                "updateTime": now,
                "updateDate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "effective": True}
        )
    except:
        print >> sys.stderr, 'exception raise when inserting/updating the score: ' +  projectIds[rowNumber]

# risk items
rankRiskItemWholeSet = dataBase.RankRiskItemWholeSet
rankRiskItemWholeList = []
collection = rankRiskItemWholeSet.find({})
for item in collection:
    for element in item['tags']:
        rankRiskItemWholeList.append(element)
riskItemList = []
for index in range(len(flattenCategoryData)):
    singleList = []
    for headerIndex in range(len(flattenCategoryHeader)):
        if flattenCategoryHeader[headerIndex] in rankRiskItemWholeList and flattenCategoryData[index][headerIndex] == 1:
            tagName = flattenCategoryHeader[headerIndex].split('_')[0]
            singleList.append(tagName)
    for headerIndex in range(len(flattenNumericalHeader)):
        if flattenNumericalHeader[headerIndex] in rankRiskItemWholeList and flattenNumericalData[index][headerIndex] == 1:
            tagName1, remain = flattenNumericalHeader[headerIndex].split('_and_')
            tagName2 = remain.split('_', 1)[0]
            if tagName1 not in singleList and tagName1 != 'financeAmount': # delete financeAmount as requested by product
                singleList.append(tagName1)
            if tagName2 not in singleList and tagName2 != 'financeAmount':
                singleList.append(tagName2)
    riskItemList.append(singleList)
# write back risk items, set model to false as well when request risk items

##################################################### begin #################################################################
if len(riskItemList) != len(itemTypes):
    print >> sys.stderr, 'risk item length is not equal to item types!'
    raise Exception('risk item length is not equal to item types!')

for rowNumber in range(len(itemTypes)):
    if itemTypes[rowNumber] in ['1001']:
        riskItemList[rowNumber] = [item for item in riskItemList[rowNumber] if item.lower().find('car') < 0]
        if itemTypes[rowNumber] == '1001':
            if 'personalIncome' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('personalIncome')
            if 'occupationYear' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('occupationYear')
            if 'occupation' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('occupation')
            if 'familyIncome' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('familyIncome')
            if 'otherLoanBalance' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('otherLoanBalance')
            if 'houseTotalAmount' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('houseTotalAmount')
            if 'carTotalAmount' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('carTotalAmount')
            if 'due' not in riskItemList[rowNumber]:
                riskItemList[rowNumber].append('due')
###############################################################

rankRiskItem = dataBase.RankRiskItem
rankRiskItemHis = dataBase.RankRiskItemHis
for rowNumber in range(len(projectIds)):
    try:
         rankRiskItem.update_one(
             {"itemId":projectIds[rowNumber]},
             {"$set":{ "appid" : appIds[rowNumber],
                        "itemId" : projectIds[rowNumber],
                        "updateTime": now,
                        "updateDate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "effective": True,
                        "tags": riskItemList[rowNumber] }},
             upsert = True
         )

         rankRiskItemHis.insert_one(
             { "appid" : appIds[rowNumber],
                        "itemId" : projectIds[rowNumber],
                        "updateTime": now,
                        "updateDate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "effective": True,
                        "tags": riskItemList[rowNumber] }
         )

    except:
        print >> sys.stderr, 'exception raise when inserting/updating the risk items: ' + projectIds[rowNumber]


# set model computed and risk item computed back to true
item = dataBase.Item
for index in range(len(projectIds)):
    try:
        projectId = projectIds[index]
        appId = appIds[index]
        item.update_one(
                {"appid": appId, "itemId": projectId},
                {
                    "$set": {"modelComputed": True,
                             "riskItemComputed": True}
                }
        )
    except:
        print >> sys.stderr, 'exception raise when updating the modelComputed/riskItemComputed tags: ' + projectIds[rowNumber]














