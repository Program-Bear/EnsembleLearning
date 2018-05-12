from tqdm import tqdm
from feature import RD
from embedding import DG,RE, LoadData, LoadTestData
from Signal import DTree, SVM, validate,SVC
from util import unify

import pandas as pd
import random
import sys
import math

def BaggingSet(Reviews, Labels):
    num = len(Reviews)
    NewReviews = []
    NewLabels = []
    
    print("Generating Bagging Set")
    for i in tqdm(range(0, num)):
        temp = random.randint(0,num-1)
        NewReviews.append(Reviews[temp])
        NewLabels.append(Labels[temp])
    return NewReviews, NewLabels

def Bagging(X, Y, TX, algorithm='SVM', num=3):
    #BX = []
    #BY = []
    BA = []
    Ans = []
    total = len(TX)
    for i in range(0, num):
        print("Translate Bagging No.%d"%(i+1))
        NX,NY = BaggingSet(X,Y)
        #BX.append(NX)
        #BY.append(NY)
        clf = None
        if (algorithm == 'SVM'):
            clf = SVM(NX, NY)
        else:
            clf = DTree(NX,NY)
        BA.append(clf.predict(TX))
    
    print("Calculate Answer")
    for i in tqdm(range(0, total)):
        base = [0,0,0]
        for j in range(0, num):
            k = BA[j][i] - (-1)
            base[k] += 1        
        Ans.append((-1) + base.index(max(base)))
    
    return Ans

def BoostingSet(Reviews, Labels, Weights):
    Weights= unify(Weights)
   
    NR = []
    NL = []
    NI = []
    size = len(Reviews)
    ep = 0.000001
    #print(Weights)
    print("Generating Boosting Set")
    while len(NI) < size:
        for i in range(0, len(Weights)):
            temp = random.uniform(0, (1.0 / size)*(1+ep))
            if (temp < Weights[i]):
                NI.append(i)
        
    random.shuffle(NI)
    
    print("BoostingSet Size: %d"%(len(NI)))
    for i in NI:
        NR.append(Reviews[i])
        NL.append(Labels[i])
    
    return NR, NL

def AdaBoost(X, Y, TX, algorithm='SVM', num=3, TreeDepth=1000):
    BA = []
    BB = []
    Ans = []
    tt = len(TX) #total test
    st = len(X) #total sample
    SW = []
    
    NX = []
    NY = []
    
    for i in X:
        NX.append(i)
    for i in Y:
        NY.append(i)
    for i in range(0, st):
        SW.append(1.0/st)
    
    belta = 1.0
    tree_depth = 1000
    
    LA = [] #last iteration's ans
    for i in range(0, st):
        LA.append(0)
    
    for i in range(0, num):
        print("Translate AdaBoost Iteration No.%d"%(i+1))
        #print(SW)
        if (algorithm == 'SVM'):
            NX,NY = BoostingSet(X, Y, SW)
        #print(SW)
        #Train Weighted Classifier
        clf = None
        if (algorithm == 'SVM'):
            clf = SVM(NX, NY)
        else:
            clf = DTree(X, Y, SW, TreeDepth)
        
        #Make a prediction
        LA = clf.predict(X)
        
        BA.append(clf.predict(TX))
        
        print("Calculate Error Rate")
        error_num = 0.0
        for j in tqdm(range(0, st)):
            if (LA[j] != Y[j]):
                error_num += 1
        
        error_rate = error_num / st
        
        print('error_rate: %s'%str(error_rate))
        if (error_rate == 0):
            print("Warning! A perfect classifier arise. abort the loop")
            num = i
            break
            

        if (error_rate > 0.5):
            print("Warning! A bad signal classifier arise. abort the loop")
            num = i
            break

        
        #Update belta
        belta = error_rate / (1 - error_rate)

        print("Update Weight")
        for j in tqdm(range(0, st)):
            if (LA[j] == Y[j]):
                #print("Update correct")
                SW[j] = SW[j] * belta
            else:
                #print("Update False")
                SW[j] = SW[j]
        
        BB.append(math.log(1 / belta))
    
    print("Calculate Answer")
    for i in tqdm(range(0, tt)):
        temp = [0,0,0]
        for j in range(0, num):
            index = BA[j][i] - (-1)
            temp[index] += BB[j]
        Ans.append((-1) + temp.index(max(temp)))

    return Ans

if __name__ == "__main__":
    dic_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    algorithm = sys.argv[4]
    n = int(sys.argv[5])
    ensemble = sys.argv[6]
    mode = sys.argv[7]
    
    sys_num = 7
    if (mode == 'test'):
        sys_num += 1
        output_file = sys.argv[sys_num]
    train_num = -1 #-1 means take all sample to train (including the validation set)
    val_num = 500
    
    if (mode == 'validation'):
        sys_num += 1
        train_num = int(sys.argv[sys_num])
        sys_num += 1
        val_num = int(sys.argv[sys_num])
    tree_depth = 1000
    
    if (algorithm == 'DTree'):
        sys_num += 1
        tree_depth = int(sys.argv[sys_num])

    ts = 0     #train set start
    te = train_num    #train set end
    
    Dic = DG(dic_path)
    
    print("Load Train Data")
    X, Y = LoadData(train_path,Dic, ts, te)
    TX = []
    TY = []
    
    if (mode == 'validation'):
        SP = te - val_num #Split Point
        assert(SP < te)
        TX = X[SP:-1]
        V_Y = Y[SP:-1]
        X = X[0:SP]
        Y = Y[0:SP]
    if (mode == 'test'):
        TX = LoadTestData(test_path, Dic)
        
    Ans = []
    if (ensemble == "Bagging"):
        Ans = Bagging(X, Y, TX, algorithm, n)
    if (ensemble == "Boosting"):
        Ans = AdaBoost(X, Y, TX, algorithm, n, tree_depth)
    
    if (mode == 'validation'):
        rate, rmse = validate(Ans, V_Y)
        #print("Answer:")
        #print(Ans)
        print("Correct Rate: %s, RMSE: %s"%(str(rate), str(rmse)))
    
    if (mode == 'test'):
        Final = []
        tot = len(Ans)
        for i in range(1, tot+1):
            Final.append([i,Ans[i - 1]])
        pd.DataFrame(Final, columns = ['id','label']).to_csv(output_file)

