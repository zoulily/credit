# coding=utf-8
import sys
sys.path.append("../Commons/Utils")
from ModelDataProcessor import ModelDataProcessor
import numpy as np
import time
from sklearn import decomposition
from sklearn.cluster import KMeans
import pickle
# import VisualizationUtility as vs
import operator
from sklearn.externals import joblib
from os import walk
class BPModelDataProcessor(ModelDataProcessor):

    def __init__(self, capFactor=4, stdThreshold=0.8, nClusters=3, nComponents=1):
        self.capFactor = capFactor
        self.stdThreshold = stdThreshold
        self.nClusters = nClusters
        self.nComponents = nComponents
        self.kmList = None
        self.sortedMap = None

    """
    get the data through the KMeans clustering process
    the first returned object is the list containing the header and the second is the actual data
    """
    def getModelNeededData(self):
        pass

    """
    get the map that storing the tags that need clustering processing
    the map is sorted is descending order by the std
    and the transform map for visualization
    """

    def getAssociatedMapFromPCA(self, input, missingValue='-9'):
        header, data = input
        # numpyData = np.array((data, missingValue), dtype=np.float)
        stdmap = {}
        transmap = {}
        nRow, nColumn = data.shape
        for index in xrange(0, nColumn):
            for indexAfter in xrange(index+1, nColumn):
                begin = time.time()
                pca = decomposition.PCA(n_components=self.nComponents)
                trans = pca.fit_transform(np.c_[data[:, index], data[:, indexAfter]])
                end = time.time()
                print 'PCA for '+ str(header[index]) + ' and ' + str(header[indexAfter]) + ' takes ' + str(end-begin)
                if pca.explained_variance_ratio_ >= self.stdThreshold:
                    ind = str(index) + '#' + str(indexAfter)
                    stdmap[ind] = pca.explained_variance_ratio_
                    transmap[ind] = trans  #  This is for the usage of visualization
        sortedMap = sorted(stdmap.items(), key=operator.itemgetter(1), reverse=True)
        self.sortedMap = sortedMap
        return sortedMap, transmap

    """
    get the KMeans algorithm list, together with the visulization figures
    """
    def getKMeansListByCalculation(self, input, mapInput, missingValue='-9', path='../Figures'):
        header, data = input
        # numpyData = np.array((data, missingValue), dtype=np.float)
        sortedMap, transmap = mapInput
        kmList = []
        tlength = 0  # ignore the id column
        for index in range(len(sortedMap)):
            begin = time.time()
            (key, value) = sortedMap[index]
            (col1, col2) = key.strip().split('#')
            intcol1 = int(col1)
            intcol2 = int(col2)
            X = data[:, intcol1]
            Y = data[:, intcol2]
            km = KMeans(n_clusters=self.nClusters, init='random', n_init=1, verbose=1)
            km.fit(np.c_[X, Y])
            tlength += km.n_clusters
            s = pickle.dumps(km)
            kmList.append(s)
            transform = transmap[key]  # get the corresponding transfrom vector
            end = time.time()
            # printToFile(('KMeans algorithm for ' + str(num_var_names[intcol1]) + ' and '+str(num_var_names[intcol2]) + ' takes '+str(end-begin)), log_file, 'a')
            # show the plotted figures, turn on when needed
            # vs.plotAltogether(X, Y, transform, km, path, ['#4EACC5', '#FF9C34', '#4E9A06'],str(header[intcol1]), str(header[intcol2]))
        self.kmList = kmList
        return self.kmList

    """
    save the KMeans model to file
    """
    def saveKMeansListToFile(self, path, header):
        kmList = self.kmList
        sortedMap = self.sortedMap
        if kmList is None or sortedMap is None:
            print 'KM List is None or SortedMap is None!'
            return
        if len(sortedMap) != len(kmList):
            print 'the length of KM List is not equal to that of SortedMap!'
            return
        for index in range(len(kmList)):
            key, value = sortedMap[index]
            (col1, col2) = key.strip().split('#')
            intcol1 = int(col1)
            intcol2 = int(col2)
            km = kmList[index]
            kmAlg = pickle.loads(km)
            tagName1 = header[intcol1]
            tagName2 = header[intcol2]
            filePath = path + tagName1 + '_' + tagName2 + '.pkl'
            joblib.dump(kmAlg, filePath)
        return

    """
    get the KMeans algorithm list from file storing the pre-calculated ones
    """
    def getKMeansListFromFile(self, path):
        files = []
        sortedMap = {}
        kmList = []
        for (dirpath, dirnames, filenames) in walk(path):
            files.extend(filenames)
            break
        for fileName in files:
            name = fileName.split('.',1)[0]
            key = name.split('_', 1)[-1]
            sortedMap[key] = None
            km = joblib.load(fileName)
            s = pickle.dumps(km)
            kmList.append(s)
        self.kmList = kmList
        self.sortedMap = sortedMap
        return kmList

    """
    generatre the matplot figures for the visualization of the KMeans clustering
    """
    def generateKMeansClusteringFigures(self, path):
        pass

    """
    get the pre-processed category data for model
    """
    def getPreProcessedFlattenCategoryData(self, input, categoryInfo, dropTags=[], missingValue=-9):
        header, data = input
        # numpyData = np.array((data, missingValue), dtype=np.int)
        retHeader = []
        retData = None
        for index in range(len(header)):
            tagName = header[index]
            if tagName in dropTags:
                continue
            dataCol = data[:, index]
            mappingInfo = self.getCategoryInfoForTag(categoryInfo, tagName)
            if mappingInfo is not None:
                lenItem = len(mappingInfo) + 1 # including the missing value
                for item in mappingInfo.items():
                    value, name = item
                    retHeader.append(tagName+'_'+str(value))
                retHeader.append(tagName+'_'+str(missingValue))
                tagDataElement = np.empty((1, lenItem), dtype=np.int32)
                tagDataElement = np.delete(tagDataElement, (0, 0), axis=0)
                for row in range(len(dataCol)):
                    offset = dataCol[row]
                    if offset == int(missingValue):
                        offset = lenItem-1
                    vector = np.zeros(lenItem, dtype=np.int32)
                    vector[offset] = 1
                    tagDataElement = np.vstack([tagDataElement, vector])
                if retData is None:
                    retData = tagDataElement
                else:
                    retData = np.c_[retData, tagDataElement]
        return retHeader, retData

    """
    get the preprocessed numerical data for model
    """
    def getPreProcessedFlattenNumericalData(self, input, dropTags=[], missingValue=-9):
        header, data = input
        # numpyData = np.array((data, missingValue), dtype=np.int)
        if self.kmList is None or self.sortedMap is None:
            print 'KM List is None or SortedMap is None!'
            return
        if len(self.sortedMap) != len(self.kmList):
            print 'the length of KM List is not equal to that of SortedMap!'
            return
        dropTagsIndices = []
        retHeader = []

        for index in range(len(header)):
            if header[index] in dropTags:
                dropTagsIndices.append(index)

        #get the length of one single row
        tlength = 0
        for j in xrange(0, len(self.sortedMap)):
            (key, value) = self.sortedMap[j]
            (col1, col2) = key.strip().split('#')
            intcol1 = int(col1)
            intcol2 = int(col2)
            if intcol1 in dropTagsIndices or intcol2 in dropTagsIndices:
                continue
            tlength += self.nClusters
            nameX = header[intcol1]
            nameY = header[intcol2]
            name_combined = nameX + '_and_' + nameY
            for j in xrange(0, self.nClusters):
                retHeader.append(name_combined + '_cluster_' + str(j))

        retData = np.empty((1, tlength), dtype=np.int32)
        retData = np.delete(retData, (0, 0), axis=0)  # delete the first row which is filled with random numbers

        for row in range(len(data)):
            singlerow = data[row, :]  # deal with each row
            output = None
            begin = time.time()
            for j in xrange(0, len(self.sortedMap)):
                (key, value) = self.sortedMap[j]
                (col1, col2) = key.strip().split('#')
                intcol1 = int(col1)
                intcol2 = int(col2)
                if intcol1 in dropTagsIndices or intcol2 in dropTagsIndices:
                    continue
                kmalg = pickle.loads(self.kmList[j])
                pre = kmalg.predict(np.c_[singlerow[intcol1], singlerow[intcol2]])
                vector = np.zeros(kmalg.n_clusters, dtype=np.int32)
                vector[pre] = 1
                if output is None:
                    output = np.array([vector])
                else:
                    output = np.c_[output, np.array([vector])]
                end = time.time()
                #printToFile('the '+str(i)+'th row takes '+str(end-begin)+' seconds for getting the KMeans clusters', log_file, 'a')
            retData = np.vstack([retData, output])
        return retHeader, retData


    """
    get the category value mapping for the
    """
    def getCategoryInfoForTag(self, categoryInfo, tag):
        for sel in categoryInfo.keys():
            tags = categoryInfo[sel]
            if tag in tags.keys():
                return tags[tag]
        return None
