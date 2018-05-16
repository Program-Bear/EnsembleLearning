from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from embedding import DG,RE,DO
from embedding import LoadData, LoadTestData
from util import MutualInfo,dic2sortedlist
import numpy as np
import sys

def RD(Raw, Label, Dim):
    raw_dim = len(Raw[0])
    
    print("Start Dimension Reduction")
    Reviews = SelectKBest(mutual_info_classif, k=Dim).fit_transform(Raw, Label)
    print("Dimension Reduction from %d to %d using chi2"%(raw_dim, Dim))
    
    return Reviews

def FD(Reverse, Feature):
    NewDic = {}
    num = len(Feature)

    for f in range(0, num):
        now = Feature[f]
        sf = str(now)
        
        key = ''
        try:
            key = Reverse[sf]
        except:
            print("Key Not Found Of %s"%sf)
        NewDic[key] = f
    
    return NewDic

def TreeBasedFeatureSelection(X, Y):
    clf = tree.DecisionTreeClassifier()
    print("Start Tree Based Feature Selection")
    clf = clf.fit(X,Y)
    print("End Tree Based Feature Sselection")
    
    ip = list(clf.feature_importances_)
    raw = len(ip)
    mean = sum(ip) / float(raw)
    features = []
    for i in range(0, raw):
        if (ip[i] > mean):
            features.append(i)
    new = len(features)
    
    print("Select %d features from %d"%(new, raw))
    #print(features)
    return features

def L1BasedFeatureSelection(X, Y):
    print("Start L1 Based Feature Selection")
    clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X,Y)
    print("End L1 Based Feature Selection")
    
    ce = list(clf.coef_)
    raw = len(ce)
    mean = sum(ce) / float(raw)
    features = []
    
    for i in range(0, raw):
        if (ce[i] > mean):
            features.append(i)
    new = len(features)

    print("Select %d features from %d"%(new, raw))
    return features

def SetUpMutualInfoDic(X, Y, r_dic):
    n_dic = {}
    f_dic = {}
    print("Start Self Defined Feature Selection")
    
    Score = list(MutualInfo(X,Y))

    print("End Self Defined Feature Selection")

    for i in range(0, len(Score)):
        n_dic[i] = Score[i]
        key = ''
        try:
            si = str(i)
            key = r_dic[si]
        except:
            print("Key not found: %d"%i)
            continue
        f_dic[key] = Score[i]

    return n_dic, f_dic

def SelfFeatureSelection(n_dic, num=2000):
    features = []
#   print("Start Self Defined Feature Selection")
    

#   print("End Self Defined Feature Selection")
    lst = dic2sortedlist(n_dic)
    raw = len(lst)
        
    for i in range(0, num):
        features.append(int(lst[i][0]))

    
    return features
    
if __name__ == "__main__":
    dic = DG('dic.txt')
    
    r_dic = DG('r_dic.txt')
    
    i_dic = DG('my_num_info.txt')
    
    #X, Y = LoadData('exp2.train.csv',dic,0,-1)

    #num = int(sys.argv[1])
    #output_dic = sys.argv[2]
    #Feature = TreeBasedFeatureSelection(X,Y)
    #Feature = L1BasedFeatureSelection(X,Y)
    num = 2000
    Feature = SelfFeatureSelection(i_dic, num)
    #print(Feature)
    
    num_dic = FD(r_dic, Feature)
    #num_dic,feature_dic = SetUpMutualInfoDic(X,Y,r_dic)

    DO(num_dic, 'my_dic.txt')

    #DO(feature_dic, 'my_feature_info.txt')
    #DO(num_dic, 'my_num_info.txt')
    
    #new_dic = FD(r_dic, [4,19,20,33])
    #print(new_dic)
    #r_dic = {}
    #for key in dic.keys():
    #    try:
    #        value = int(dic[key])
    #    except:
    #        continue
    #    r_dic[value] = key
    #DO(r_dic, 'r_dic.txt')
    
    #file_object = open("r_dic.txt",'w')
    #for key in r_dic.keys():
    #    value = r_dic[key]
    #    s = str(key) + ":" + str(value) + '\n'
    #    file_object.write(s)
    #file_object.close()
    
