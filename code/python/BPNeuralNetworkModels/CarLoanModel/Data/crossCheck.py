# coding=utf-8

def crossCheck(fileParent, fileChild):
    headerParent = []
    dataParent = []
    headerChild = []
    dataChild = []
    with open(fileParent) as f:
        line = f.readline()
        headerParent = line.strip().split(',')
        for line in f:
            dataParent.append(line.strip().split(','))
    with open(fileChild) as f:
        line = f.readline()
        headerChild = line.strip().split(',')
        for line in f:
            dataChild.append(line.strip().split(','))
    if len(dataChild) != len(dataParent):
        print 'row numbers in child is not equal to that in parent'
    for indexChild in range(len(headerChild)):
        tag = headerChild[indexChild]
        indexParent = headerParent.index(tag)
        for rowIndex in range(len(dataChild)):
            if dataParent[rowIndex][indexParent] != dataChild[rowIndex][indexChild]:
                print rowIndex, ' parent: ', dataParent[rowIndex][indexParent], 'child: ', dataChild[rowIndex][indexChild]

crossCheck('FlattenData.csv', 'replaceMissingNumericalData.csv')

