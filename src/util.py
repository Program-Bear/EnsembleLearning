import math
import numpy as np
from tqdm import tqdm
def unify(S):
    NS = []
    num = len(S)
    tot = float(sum(S))
    for i in range(0, num):
        NS.append(S[i] / tot)
    #print(NS)
    return NS

def MyMutul(XY,X,Y):
    return math.log(float(XY) / (X * Y))

def MutualInfo(X,Y):
    SampleCount = len(X)
    FeatureCount = len(X[0])
    ClassCount = 3
    assert(len(Y) == SampleCount)
    
    FeatureClass = np.ones([FeatureCount, ClassCount])
    Class = np.zeros([ClassCount])
    Feature = np.zeros([FeatureCount])
    
    #Calculate Class Num
    print("Calculate Class Num")
    for i in tqdm(range(0, len(Y))):
        Class[Y[i] - (-1)] += 1

    #Calculate Feature Num
    print("Calculate Feature Num and Class Feature Num")
    for i in tqdm(range(0, len(X))):
        Sample = X[i]
        Label = Y[i] - (-1)
        assert(len(Sample) == FeatureCount)
        for j in range(0, FeatureCount):
            Feature[j] += Sample[j]
            FeatureClass[j][Label] += Sample[j]
    
    FeatureInfo = np.zeros([FeatureCount])
    #Calculate Feature Info
    print("Calculate Feature Info")
    MIN = -100000000
    for i in tqdm(range(0, FeatureCount)):
        if (Feature[i] < 10):
            FeatureInfo[i] = MIN
            continue
        if (Feature[i] > SampleCount * 0.9):
            FeatureInfo[i] = MIN
            continue
        
        for j in range(0, ClassCount):
            FeatureInfo[i] += (FeatureClass[i][j]) * MyMutul(FeatureClass[i][j], Feature[i], Class[j])

    return FeatureInfo
    
def dic2sortedlist(dic, Reverse=True):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key,val) for key, val in zip(keys,vals)]
    lst = sorted(lst, key = lambda x:float(x[1]), reverse = Reverse)
    return lst
