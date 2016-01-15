# coding=utf-8
import sys
sys.path.append("../../Commons/DataProcessor")

from MongoDataProcessor import MongoDataProcessor
mongoDataProcessorTest = MongoDataProcessor('1019', address='192.168.1.120')

'''
#Test for getMetaTagList
allTags = mongoDataProcessorTest.getMetaTagList()
print allTags
'''


#Test for getCategoryInfo
categoryInfo = mongoDataProcessorTest.getCategoryInfo()
for multiSelect in categoryInfo.keys():
    allBinaryTags = categoryInfo[multiSelect]
    print str(multiSelect) + ':'
    indentChar = '\t'
    for tag in allBinaryTags.keys():
        valueGroup = allBinaryTags[tag]
        print indentChar+tag+':'
        for value in valueGroup.keys():
            name = valueGroup[value]
            print indentChar+'\t',value,' '+name


'''
#Test for getFlattenTagData
flattenHeader, flattenData = mongoDataProcessorTest.getFlattenTagData(additionalTags=['creditScore'])
with open('Data/FlattenData.csv', 'w') as f:
    f.write(','.join(flattenHeader)+'\n')
    for dataRow in flattenData:
        print >> f,  ','.join(dataRow)#.encode('utf-8')
'''

'''
targetHeader, targetData = mongoDataProcessorTest.getTargetTagData(targetTags=['creditScore'])
print ','.join(targetHeader)
for item in targetData:
    print ','.join(item)

'''

'''
categoryHeader, categoryData = mongoDataProcessorTest.getFlattenCategoryData()
numericalHeader, numericalData = mongoDataProcessorTest.getFlattenNumericalData()
with open('Data/FlattenDataCategory.csv', 'w') as f:
    f.write(','.join(categoryHeader)+'\n')
    for dataRow in categoryData:
        f.write(','.join(dataRow)+'\n')
with open('Data/FlattenDataNumerical.csv', 'w') as f:
    f.write(','.join(numericalHeader)+'\n')
    for dataRow in numericalData:
        f.write(','.join(dataRow)+'\n')
'''

'''
groupedCategoryData = mongoDataProcessorTest.getGroupedCategoryData()
for tagGroupName in groupedCategoryData.keys():
    with open('Data/GroupedCategoryInfo_'+tagGroupName+'.csv', 'w') as f:
        header, data = groupedCategoryData[tagGroupName]
        f.write(','.join(header)+'\n')
        for rawData in data:
            f.write(','.join(rawData)+'\n')

groupedNumericalData = mongoDataProcessorTest.getGroupedNumericalData()
for tagGroupName in groupedNumericalData.keys():
    with open('Data/GroupedNumericalInfo_'+tagGroupName+'.csv', 'w') as f:
        header, data = groupedNumericalData[tagGroupName]
        f.write(','.join(header)+'\n')
        for rawData in data:
            f.write(','.join(rawData)+'\n')
'''
"""
tagNamesByTagGroup = mongoDataProcessorTest.getTagNamesByTagGroups()
for key in tagNamesByTagGroup:
    print key
    print tagNamesByTagGroup[key]
"""

'''
tagDataByTagGroups = mongoDataProcessorTest.getTagDataByTagGroups(additionalTags=['creditScore'])
for tagGroupName in tagDataByTagGroups.keys():
    tagGroup = tagDataByTagGroups[tagGroupName]
    with open('Data/TagGroupData_'+tagGroupName+'.csv', 'w') as f:
        for rowData in tagGroup:
            f.write(','.join(rowData)+'\n')
'''



