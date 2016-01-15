# coding=utf-8

import time
import sys

import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
sys.path.append("../Commons/Model")

from ModelBase import ModelBase


class BPModelTrainer(ModelBase):

    def __init__(self,
                 nhiddenNerons=20,
                 flattenCategoryHeader=None,
                 flattenCategoryData=None,
                 flattenNumericalHeader=None,
                 flattenNumericalData=None,
                 flattenTargetHeader=None,
                 flattenTargetData=None):
        # super(BPModelTrainer, self).__init__()
        self.flattenCategoryHeader = flattenCategoryHeader
        self.flattenCategoryData = flattenCategoryData
        self.flattenNumericalHeader = flattenNumericalHeader
        self.flattenNumericalData = flattenNumericalData
        self.flattenTargetHeader = flattenTargetHeader
        self.flattenTargetData = flattenTargetData
        self.flattenTargetDataConverted = np.copy(flattenTargetData)
        self.net = None
        self.trainer = None
        self.nhiddenNerons = nhiddenNerons
        self.testDataSet = None
        self.trainDataSet = None
        self.finalDataSet = None
        self.finalHeaderSet = None
        self.nattributes = 0
        self.noutput = 4
        self.nbClasses = 4
        self.targetDistribution = {}
        self.riskTag = []

    def targetTransform(self):
        for row in range(len(self.flattenTargetDataConverted)):
            if self.flattenTargetDataConverted[row] >= 180:
                self.flattenTargetDataConverted[row] = 3
            elif self.flattenTargetDataConverted[row] >= 160:
                self.flattenTargetDataConverted[row] = 2
            elif self.flattenTargetDataConverted[row] >= 150:
                self.flattenTargetDataConverted[row] = 1
            else:
                self.flattenTargetDataConverted[row] = 0
        return self.flattenTargetDataConverted

    def trainModel(self):
        self.finalDataSet = np.c_[self.flattenNumericalData, self.flattenCategoryData, self.flattenTargetDataConverted]
        self.finalHeaderSet = self.flattenNumericalHeader + self.flattenCategoryHeader + self.flattenTargetHeader
        self.nattributes = self.flattenNumericalData.shape[1] + self.flattenCategoryData.shape[1]
        ds = ClassificationDataSet(self.nattributes, 1, nb_classes=self.nbClasses)
        for rowData in self.finalDataSet:
            target = rowData[-1]
            variables = rowData[0:-1]
            ds.addSample(variables, target)
        self.testDataSet, self.trainDataSet = ds.splitWithProportion(0.25)
        self.testDataSet._convertToOneOfMany()
        self.trainDataSet._convertToOneOfMany()
        print self.testDataSet
        print self.trainDataSet
        self.net = buildNetwork(self.nattributes, self.nhiddenNerons, self.noutput, hiddenclass=TanhLayer, outclass=SigmoidLayer, bias=True)
        self.trainer = BackpropTrainer(self.net, self.trainDataSet, learningrate=0.001, momentum=0.99)
        begin0 = time.time()
        # self.trainer.trainUntilConvergence(verbose=True, dataset=ds, validationProportion=0.25, maxEpochs=10)
        for i in xrange(10):
            begin = time.time()
            self.trainer.trainEpochs(10)
            end = time.time()
            print 'iteration ', i, ' takes ', end-begin,  'seconds'
        end0 = time.time()
        print 'total time consumed: ', end0 - begin0

    def saveModelToFile(self, path):
        NetworkWriter.writeToFile(self.net, path)

    def readModelFromFile(self, path):
        self.net = NetworkReader.readFrom(path)
        return self.net

    def getAlignedScore(self, result, coefficient=[0,150,160,180,300]): # max score is 300
        score = 0.0
        if len(result)+1 != len(coefficient):
            print ('length not equal when calculating score')
        else:
            #for e in range(len(result)):
            #    score += result[e]*coefficient[e]
            cat =  np.argmax(result)
            score = coefficient[cat] + result[cat] * (coefficient[cat+1] - coefficient[cat])
        return score

    def validateModel(self):
        trnresult = percentError(self.trainer.testOnClassData(dataset=self.trainDataSet), self.trainDataSet['class'])
        tstresult = percentError(self.trainer.testOnClassData(dataset=self.testDataSet), self.testDataSet['class'])
        print "epoch: %4d" % self.trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult

        attributeSet = np.c_[self.flattenNumericalData, self.flattenCategoryData]
        for row in range(len(attributeSet)):
            rowData = attributeSet[row]
            tar = self.net.activate(rowData)
            score = self.getAlignedScore(tar)
            print tar, ': ', np.argmax(tar), ': ', self.flattenTargetDataConverted[row], ': ', score, ': ', self.flattenTargetData[row], ': ', np.abs(score-self.flattenTargetData[row])*1.0 / self.flattenTargetData[row]

    def generateRiskDistributionByRiskTags(self):
        offset = 3
        if len(self.flattenNumericalHeader) % 3 != 0:
            raise Exception('flatten numerical header is not dividable by 3!')
        tagGroupDistribution = {}
        index = 0
        while index < len(self.flattenNumericalHeader):
            underScoreIndex = self.flattenNumericalHeader[index].rfind('_')
            tagName = self.flattenNumericalHeader[index][0:underScoreIndex]
            clusterDistribution = {}
            for k in range(offset):
                groupName = self.flattenNumericalHeader[index+k]
                targetDistribution = {}
                for row in range(len(self.flattenNumericalData)):
                    flag = self.flattenNumericalData[row][index+k]
                    result = self.flattenTargetDataConverted[row]
                    if not targetDistribution.has_key(result):
                        targetDistribution[result] = flag
                    else:
                        targetDistribution[result] += flag
                clusterDistribution[groupName] = targetDistribution
            tagGroupDistribution[tagName] = clusterDistribution
            index += offset
        index = 0
        while index < len(self.flattenCategoryHeader):
            firstOne = self.flattenCategoryHeader[index]
            mainName = firstOne[0:firstOne.rfind('_')]
            indexLater = index + 1
            while indexLater < len(self.flattenCategoryHeader) and self.flattenCategoryHeader[indexLater].find(mainName) >= 0:
                indexLater += 1
            offset = indexLater - index
            clusterDistribution = {}
            for k in range(offset):
                groupName = self.flattenCategoryHeader[index+k]
                targetDistribution = {}
                for row in range(len(self.flattenCategoryData)):
                    flag = self.flattenCategoryData[row][index+k]
                    result = self.flattenTargetDataConverted[row]
                    if not targetDistribution.has_key(result):
                        targetDistribution[result] = flag
                    else:
                        targetDistribution[result] += flag
                clusterDistribution[groupName] = targetDistribution
            tagGroupDistribution[mainName] = clusterDistribution
            index += offset
        self.targetDistribution = tagGroupDistribution
        return self.targetDistribution

    # group 0+1 more than 66.7% while the cluster occupies more than half of the total
    def getRiskItems(self):
        nRow = len(self.flattenTargetData)
        tagsCombined = []
        # tagsList = []
        targetDistribution = self.targetDistribution
        for tagGroup in targetDistribution.keys():
            value = targetDistribution[tagGroup]
            for cluster in value.keys():
                subsum = 0
                lowRatingsum = 0
                dataMap = value[cluster]
                for key in dataMap.keys():
                    subsum += dataMap[key]
                    if key == 0 or key == 1:
                        lowRatingsum += dataMap[key]
                if subsum*1.0 / nRow > 0.5 and lowRatingsum*1.000 / subsum > 0.5: # definition for risk items
                    tagsCombined.append(cluster)
                    break
        # tagsSet = set(tagsCombined)
        '''
        for s in tagsSet:
            if s.find('_and_') >= 0:
                tag1, tag2 = s.split('_and_')
                if tag1 not in tagsList:
                    tagsList.append(tag1)
                if tag2 not in tagsList:
                    tagsList.append(tag2)
            else:
                if s not in tagsList:
                    tagsList.append(s)
        '''
        self.riskTag = tagsCombined
        return self.riskTag