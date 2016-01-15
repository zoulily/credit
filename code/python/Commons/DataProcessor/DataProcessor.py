# coding=utf-8

from abc import ABCMeta, abstractmethod

class DataProcessor:
    __metaclass__ = ABCMeta
    def __init__(self, loanType):
        self.loanType = loanType

    """
    get the list containing all the tag names
    """
    @abstractmethod
    def getMetaTagList(self): pass

    """
    get the list containing all the tag group names
    """
    @abstractmethod
    def getTagGroupNames(self): pass

    """
    get the tag names grouped by tag groups
    source: data source
    useless: tag name list containing the tags that you want to remove
    """
    @abstractmethod
    def getTagNamesByTagGroups(self, uselessTags=[]): pass

    """
    get the tag data grouped by tag groups
    useless: tag name list containing the tags that you want to remove
    onlyExists: False or True, if turned on, which is 1, then the tag names which listed in tag definition but not in data source will be excluded
    missingValue: the value that you want to replace the missing value
    """
    @abstractmethod
    def getTagDataByTagGroups(self, uselessTags=[], onlyExists=True, missingValue='-9'): pass

    """
    get the flatten tag data
    useless: tag name list containing the tags that you want to remove
    onlyExists: False or True, if turned on, which is 1, then the tag names which listed in tag definition but not in data source will be excluded
    missingValue: the value that you want to replace the missing value
    """

    @abstractmethod
    def getFlattenTagData(self, uselessTags=[], onlyExists=True, missingValue='-9', additionalTags=[]): pass

    """
    get the target header and data , first returned object is the header then the data
    """
    @abstractmethod
    def getTargetTagData(self, targetTags=[]): pass

    """
    get the flatten category data, first returned object is the header then the data
    """
    @abstractmethod
    def getFlattenCategoryData(self): pass

    """
    get the flatten numerical data, first returned object is the header then the data
    """
    @abstractmethod
    def getFlattenNumericalData(self): pass

    """
    get the category data grouped by tag groups
    """
    @abstractmethod
    def getGroupedCategoryData(self): pass

    """
    get the numerical data grouped by tag groups
    """
    @abstractmethod
    def getGroupedNumericalData(self): pass

    """
    get categorical tag information
    eg. {multiselect:{tag:{value:name}}}
    """
    @abstractmethod
    def getCategoryInfo(self): pass

    """
    get project ids
    """
    @abstractmethod
    def getProjectId(self): pass





