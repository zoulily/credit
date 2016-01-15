from __future__ import division

import numpy as np
import numpy.ma as ma
import math

"""
replace strange data with default value
return is the string type array but can be converted into integers
"""
def getFixedData(input, missingValue='-9'):
    data = input
    for dataRow in data:
        for index in range(len(dataRow)):
            try:
                item = dataRow[index]
                num = float(item)
                if num < 0 or np.isnan(num):
                    dataRow[index] = missingValue
            except Exception as e:
                dataRow[index] = missingValue
    return input

"""
delete columns with constant value
the criteria is that if the std of one column is fall below
some threshold, then it will be treated as constant
"""
def deleteColumnWithConstantValue(input, stdThreshold=0.1):
    header, data = input
    retHeader = []
    retData = None
    for index in range(len(header)):
        colData = data[:, index]
        std = colData.std()
        if std >= stdThreshold:
            if retData is None:
                retData = colData
            else:
                retData = np.c_[retData, colData]
            retHeader.append(header[index])
    return retHeader, retData

"""
replace missing value with mean value
the data should already be in numpy array format
"""
def replaceMissingWithMean(input, missingValue=-9):
    header, data = input
    statMap = {}
    row, column = data.shape
    for c in range(column):
        colData = ma.masked_values(data[:, c], missingValue)
        tagName = header[c]
        realData = data[:, c]
        std = colData.std()
        mean = colData.mean()
        max = colData.max()
        min = colData.min()
        if np.absolute(std) >= 0 and np.absolute(mean) >= 0:
            statMap[tagName] = (max, min, mean, std)
            for r in range(row):
                if realData[r] == missingValue:
                    realData[r] = mean
                    if math.isnan(realData[r]):
                        realData[r] = missingValue
    return data, statMap

# input:data
# replace missingValue with mean value
# return:data
def replaceMissingValueWithMean(data, missingValue=-9):
    row, column = data.shape
    for c in range(column):
        mean = data[:,c].mean()
        for r in range(row):
            if(data[r,c] == missingValue):
                data[r,c] = mean;
    return data

"""
print information to log file
"""
def printToFile(string, path, mode='a'):
    with open(path, mode) as f:
        f.write(string + '\n')

'''
woe transformation,note that zhe input data value should be discreted number

header:data header
data: data value
targetHeader:target header
targetData: target data
threshold:target's threshold for binary class
'''
def transWOE(header,data,targetHeader,targetData,threshold):
    badCount = goodCount = 0
    for r in targetData:
        if (r<threshold):
            badCount += 1
        else:
            goodCount += 1
    row,col = data.shape
    if(goodCount==0 or badCount==0):
        raise Exception('good or bad counts 0')
    badCountDic = {}
    goodCountDic = {}

    for i in xrange(row):
        for j in xrange(col):
            if not (badCountDic.has_key(header[j])):
                badCountDic[header[j]] = {}
            if not (goodCountDic.has_key(header[j])):
                goodCountDic[header[j]] = {}
            if(targetData[i]<threshold):
                if(badCountDic[header[j]].has_key(data[i,j])):
                    badCountDic[header[j]][data[i,j]] += 1
                else:
                    badCountDic[header[j]][data[i,j]] = 1
            else:
                if(goodCountDic[header[j]].has_key(data[i,j])):
                    goodCountDic[header[j]][data[i,j]] += 1
                else:
                    goodCountDic[header[j]][data[i,j]] = 1

    resultDic = {}
    woeData = np.ndarray(shape = (row,col), dtype = float)
    for i in xrange(row):
        for j in xrange(col):
            if not(resultDic.has_key(header[j])):
                resultDic[header[j]] = {}
            if not (resultDic[header[j]].has_key(data[i,j])):
                #set 0.5 for 0
                if not(goodCountDic[header[j]].has_key(data[i,j])):
                    goodCountDic[header[j]][data[i,j]] = 0.5
                if not(badCountDic[header[j]].has_key(data[i,j])):
                    badCountDic[header[j]][data[i,j]] = 0.5
                resultDic[header[j]][data[i,j]] = np.log((badCountDic[header[j]][data[i,j]]/goodCountDic[header[j]][data[i,j]])/(badCount/goodCount))
            woeData[i,j] = resultDic[header[j]][data[i,j]]

    return woeData,resultDic
