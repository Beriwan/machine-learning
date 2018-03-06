from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

    
def classify0(inX, dataSet, labels,k):
    dataSetSize = dataSet.shape[0]    #取dataSet行数
    diffMat = tile(inX,(dataSetSize,1)) - dataSet    #tile 重复inX 4行
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1) #axis = 1表示列相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #argsort将元素由小到大排序
    classCount = {} #dict
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    soredClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)
    #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    return soredClassCount[0][0]