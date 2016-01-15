# coding=utf-8

import json

from pymongo import MongoClient

from DataProcessor import DataProcessor


class MongoDataProcessor(DataProcessor):

    def __init__(self, loanType, source=None, address=None, normalizer=None):
        super(MongoDataProcessor, self).__init__(loanType)
        self.source = source
        self.address = address
        self.normalizer = normalizer
        self.allTags = None
        self.allTagGroupNames = None
        self.tagNamesByTagGroupsMap = None
        self.tagDataByTagGroupsMap = None
        self.flattenTagHeader = None
        self.flattenTagData = None
        self.categoryInfo = None
        self.targetHeader = None
        self.targetData = None
        self.flattenCategoryHeader = None
        self.flattenCategoryData = None
        self.flattenNumericalHeader = None
        self.flattenNumericalData = None
        self.groupedCategoryData = None
        self.groupedNumericalData = None
        self.projectIds = None
        self.appIds = None
        self.itemTypes = None # need to be deleted
        self.modelComputed = None
        self.riskItemComputed = None

    def getConnection(self):
        if self.source is None:
            self.source = MongoClient(self.address, 27017)
        return self.source

    '''
    get all the tag names for the item type
    '''
    def getMetaTagList(self):
        if self.allTags is not None:
            return self.allTags
        source = self.getConnection()
        dataBase = source.RANK
        collection = dataBase.MetaTag
        allTagsTmp = collection.find({}, {'itemType':1, 'tagKey':1, '_id':0})
        allTags = []
        for tagKey in allTagsTmp:
            if tagKey['itemType'].strip() in self.loanType:
                allTags.append(tagKey['tagKey'].encode('ascii')) #  remove the 'u' character
        self.allTags = allTags
        return self.allTags

    '''
    get all the tag group names for the item type
    '''
    def getTagGroupNames(self):
        if self.allTagGroupNames is not None:
            return self.allTagGroupNames
        source = self.getConnection()
        dataBase = source.RANK
        collection = dataBase.MetaTag
        tagGroupNames = collection.find({}, {'tagGroupName':1, '_id':0})
        allTagGroupNames = []
        for tagGroupName in tagGroupNames:
            allTagGroupNames.append(tagGroupName)
        self.allTagGroupNames = allTagGroupNames
        return self.allTagGroupNames

    '''
    get the tag names classified by the tag group names, excluding the tags specified in the uselessTags
    '''
    def getTagNamesByTagGroups(self, uselessTags=[]):
        if self.tagNamesByTagGroupsMap is not None:
            return self.tagNamesByTagGroupsMap
        source = self.getConnection()
        dataBase = source.RANK
        collection = dataBase.MetaTag
        tagGroupNameAndTag = collection.find({}, {'itemType':1, 'tagGroupName':1, 'tagKey':1, '_id':0})
        tagNamesByTagGroupsMap = {}
        for item in tagGroupNameAndTag:
            if item['itemType'] not in self.loanType:
                continue
            key = item['tagGroupName']
            value = item['tagKey']
            if key in uselessTags:
                continue
            if not tagNamesByTagGroupsMap.has_key(key):
                tagList = []
                tagList.append(value)
                tagNamesByTagGroupsMap[key] = tagList
            else:
                tagNamesByTagGroupsMap[key].append(value)
        self.tagNamesByTagGroupsMap = tagNamesByTagGroupsMap
        return self.tagNamesByTagGroupsMap

    '''
    get the tag data classified by tag groups, the first line is the tag name
    '''
    def getTagDataByTagGroups(self, uselessTags=[], onlyExists=True, missingValue='-9', additionalTags=[]):
        if self.tagDataByTagGroupsMap is not None:
            return self.tagDataByTagGroupsMap
        source = self.getConnection()
        dataBase = source.RANK
        collection = dataBase.MetaTag
        header, data = self.getFlattenTagData(uselessTags, onlyExists, missingValue, additionalTags)
        tagNamesByTagGroupsMap = self.getTagNamesByTagGroups(uselessTags)
        tagDataByTagGroupsMap = {}
        tagHeaderByTagGroupsMap = {}
        others = '999999'
        for tagGroupName in tagNamesByTagGroupsMap.keys():
            names = tagNamesByTagGroupsMap[tagGroupName]
            dataIndex = []
            tagNamesInGroup = []
            for tagName in names:
                if tagName in header:
                    index = header.index(tagName)
                    dataIndex.append(index)
                    tagNamesInGroup.append(tagName)
            dataCollection = []
            dataCollection.append(tagNamesInGroup)
            for row in data:
                dataRow = []
                for index in dataIndex:
                    dataRow.append(row[index])
                dataCollection.append(dataRow)
            tagDataByTagGroupsMap[tagGroupName] = dataCollection

        # additionalTags
        additionalDataCollection = []
        additionalDataIndex = []
        additionalTagNamesInGroup = []
        for name in additionalTags:
            if name in header:
                index = header.index(name)
                additionalDataIndex.append(index)
                additionalTagNamesInGroup.append(name)
        additionalDataCollection.append(additionalTagNamesInGroup)
        for row in data:
            additionalData = []
            for index in additionalDataIndex:
                additionalData.append(row[index])
            additionalDataCollection.append(additionalData)
        tagDataByTagGroupsMap[others] = additionalDataCollection

        self.tagDataByTagGroupsMap = tagDataByTagGroupsMap
        return self.tagDataByTagGroupsMap

    """
    get the flatten tag data
    source: data source
    useless: tag name list containing the tags that you want to remove
    onlyExists: False or True, if turned on, which is 1, then the tag names which listed in tag definition but not in data source will be excluded
    missingValue: the value that you want to replace the missing value
    """
    def getFlattenTagData(self, uselessTags=[], onlyExists=True, missingValue='-9', additionalTags=[], computeNeeded=False, fromTraining=True):
        if self.flattenTagData is not None and self.flattenTagHeader is not None:
            return self.flattenTagHeader, self.flattenTagData
        source = self.getConnection()
        dataBase = source.RANK
        if fromTraining:
            collection = dataBase.TrainingSetCarLoan
        else:
            collection = dataBase.Item
        if not computeNeeded:
            tags = collection.find({}, {'itemType':1, 'tags':1, '_id':0, 'itemId':1, 'appid':1, 'modelComputed':1, 'riskItemComputed':1})
        else:
            tags = collection.find({"$or":[{"modelComputed":False}, {"riskItemComputed":False}]}, {'itemType':1, 'tags':1, '_id':0, 'itemId':1, 'appid':1, 'modelComputed':1, 'riskItemComputed':1})
        headerSet = set()
        flattenTagHeader = []
        flattenTagData = []
        projectIds = []
        appIds = []
        itemTypes = []
        modelComputed = []
        riskItemComputed = []
        metaTags = self.getMetaTagList()
        # get the header first
        if onlyExists:
            for tag in tags:
                if tag['itemType'] not in self.loanType:
                    continue
                jsonMap = json.loads(tag['tags'])
                for key in jsonMap:
                    if key not in uselessTags:
                        headerSet.add(key)
            for name in headerSet:
                if name in metaTags or name in additionalTags:
                    flattenTagHeader.append(name)
        else:
            flattenTagHeader = metaTags + additionalTags
        tags.rewind() # cursor rewind
        for tag in tags:
            if tag['itemType'] not in self.loanType:
                continue
            if tag.has_key('itemId'):
                projectIds.append(tag['itemId'])
            else:
                projectIds.append('')
            if tag.has_key('appid'):
                appIds.append(tag['appid'])
            else:
                appIds.append('')
            if tag.has_key('itemType'):
                itemTypes.append(tag['itemType'])
            else:
                itemTypes.append('')
            if tag.has_key('modelComputed'):
                modelComputed.append(tag['modelComputed'])
            else:
                modelComputed.append(True)
            if tag.has_key('riskItemComputed'):
                riskItemComputed.append(tag['riskItemComputed'])
            else:
                riskItemComputed.append(True)
            dataRow = []
            jsonMap = json.loads(tag['tags'])
            for tagName in flattenTagHeader:
                if jsonMap.has_key(tagName):
                    value = jsonMap[tagName].encode('utf-8')  # support chinese character
                    if value.strip() is '' or value.strip() == 'null':
                        value = missingValue
                else:
                    value = missingValue
                dataRow.append(value)
            flattenTagData.append(dataRow)
        self.flattenTagHeader = flattenTagHeader
        self.flattenTagData = flattenTagData  # convert to numpy format
        self.projectIds = projectIds
        self.appIds = appIds
        self.itemTypes = itemTypes
        self.modelComputed = modelComputed
        self.riskItemComputed = riskItemComputed
        return self.flattenTagHeader, self.flattenTagData

    '''
    header + data
    '''
    def getTargetTagData(self, targetTags=[]):
        if self.targetHeader is not None and self.targetData is not None:
            return self.targetHeader, self.targetData
        if self.flattenTagData is None and self.flattenTagHeader is None:
            raise Exception('Please generate flatten data first!')
        targetHeader = targetTags
        targetData = []
        targetIndex = []
        for target in targetTags:
            index = self.flattenTagHeader.index(target)
            targetIndex.append(index)
        for data in self.flattenTagData:
            rowData = []
            for index in targetIndex:
                rowData.append(data[index])
            targetData.append(rowData)
        self.targetHeader = targetHeader
        self.targetData = targetData
        return self.targetHeader, self.targetData

    def getFlattenCategoryData(self):
        if self.flattenCategoryHeader is not None and self.flattenCategoryData is not None:
            return self.flattenCategoryHeader, self.flattenCategoryData
        if self.flattenTagData is None and self.flattenTagHeader is None:
            raise Exception('Please generate flatten data first!')
        flattenCategoryHeader = []
        flattenCategoryData = []
        flattenCategoryIndex = []
        for tagName in self.flattenTagHeader:
            if self.isCategoryTag(tagName) and tagName in self.getMetaTagList(): # not include the target tag
                flattenCategoryHeader.append(tagName)
                index = self.flattenTagHeader.index(tagName)
                flattenCategoryIndex.append(index)
        for data in self.flattenTagData:
            rowData = []
            for index in flattenCategoryIndex:
                rowData.append(data[index])
            flattenCategoryData.append(rowData)
        self.flattenCategoryHeader = flattenCategoryHeader
        self.flattenCategoryData = flattenCategoryData
        return self.flattenCategoryHeader, self.flattenCategoryData

    def getFlattenNumericalData(self):
        if self.flattenNumericalHeader is not None and self.flattenNumericalData is not None:
            return self.flattenNumericalHeader, self.flattenNumericalData
        if self.flattenTagData is None and self.flattenTagHeader is None:
            raise Exception('Please generate flatten data first!')
        flattenNumericalHeader = []
        flattenNumericalData = []
        flattenNumericalIndex = []
        for tagName in self.flattenTagHeader:
            if not self.isCategoryTag(tagName) and tagName in self.getMetaTagList(): # numerical
                flattenNumericalHeader.append(tagName)
                index = self.flattenTagHeader.index(tagName)
                flattenNumericalIndex.append(index)
        for data in self.flattenTagData:
            rowData = []
            for index in flattenNumericalIndex:
                rowData.append(data[index])
            flattenNumericalData.append(rowData)
        self.flattenNumericalHeader = flattenNumericalHeader
        self.flattenNumericalData = flattenNumericalData
        return self.flattenNumericalHeader, self.flattenNumericalData

    """
    get categorical tag information
    eg. {multiselect:
                        {
                         tag1:{value1:name1, value2:name2, value3:name3},
                         tag2:{value1:name1, value2:name2, value3:name3}
                        }
        }
    """
    def getCategoryInfo(self):
        if self.categoryInfo is not None:
            return self.categoryInfo
        source = self.getConnection()
        dataBase = source.RANK
        collection = dataBase.MetaTagValueDef
        categoryTagsTmp = collection.find({}, {'_id':0})
        categoryInfo = {}
        for unit in categoryTagsTmp:
            tagKey = unit['tagKey']
            multiSelect = unit['multiSelect']
            value = unit['value']
            name = unit['name']
            if categoryInfo.has_key(multiSelect):
                allTagsInfo = categoryInfo[multiSelect]
                if allTagsInfo.has_key(tagKey):
                    valueMap = allTagsInfo[tagKey]
                    valueMap[value] = name
                else:
                    allTagsInfo[tagKey] = {value: name}
            else:
                allTagsInfo = {}
                allTagsInfo[tagKey] = {value: name}
                categoryInfo[multiSelect] = allTagsInfo
        self.categoryInfo = categoryInfo
        return self.categoryInfo

    def isCategoryTag(self, tag):
        categoryInfo = self.getCategoryInfo()
        for multiSelect in categoryInfo.keys():
            tagsMap = categoryInfo[multiSelect]
            if tag in tagsMap.keys():
                return True
        return False

    """
    get the category data grouped by tag groups
    """
    def getGroupedCategoryData(self):
        if self.groupedCategoryData is not None:
            return self.groupedCategoryData
        tagNamesByTagGroups = self.getTagNamesByTagGroups()
        groupedCategoryData = {}
        flattenCategoryHead, flattenCategoryData = self.getFlattenCategoryData()
        for tagGroupName in tagNamesByTagGroups.keys():
            tagsInGroup = tagNamesByTagGroups[tagGroupName]
            tagIndices = []
            tagHeaders = []
            tagData = []
            for tag in tagsInGroup:
                if tag in flattenCategoryHead:
                    tagIndices.append(flattenCategoryHead.index(tag))
                    tagHeaders.append(tag)
            for row in flattenCategoryData:
                dataRow = []
                for index in tagIndices:
                    dataRow.append(row[index])
                tagData.append(dataRow)
            groupedCategoryData[tagGroupName] = (tagHeaders, tagData)
        self.groupedCategoryData = groupedCategoryData
        return self.groupedCategoryData

    """
    get the numerical data grouped by tag groups
    """
    def getGroupedNumericalData(self): 
        if self.groupedNumericalData is not None:
            return self.groupedNumericalData
        tagNamesByTagGroups = self.getTagNamesByTagGroups()
        groupedNumericalData = {}
        flattenNumericalHead, flattenNumericalData = self.getFlattenNumericalData()
        for tagGroupName in tagNamesByTagGroups.keys():
            tagsInGroup = tagNamesByTagGroups[tagGroupName]
            tagIndices = []
            tagHeaders = []
            tagData = []
            for tag in tagsInGroup:
                if tag in flattenNumericalHead:
                    tagIndices.append(flattenNumericalHead.index(tag))
                    tagHeaders.append(tag)
            for row in flattenNumericalData:
                dataRow = []
                for index in tagIndices:
                    dataRow.append(row[index])
                tagData.append(dataRow)
            groupedNumericalData[tagGroupName] = (tagHeaders, tagData)
        self.groupedNumericalData = groupedNumericalData
        return self.groupedNumericalData

    def getProjectId(self):
        return self.projectIds

    def getAppId(self):
        return self.appIds

    def getItemType(self):
        return self.itemTypes

    def getModelComputed(self):
        return self.modelComputed

    def getRiskItemComputed(self):
        return self.riskItemComputed