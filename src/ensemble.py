from tqdm import tqdm
from feature import RD
from embedding import DG,RE, LoadData, LoadTestData
from Signal import DTree, SVM, validate,SVC, KNN
from util import unify

import argparse
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
        if (algorithm == 'DTree'):
            clf = DTree(NX,NY)
        if (algorithm == 'KNN'):
            clf = KNN(NX,NY)
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
    max_size = len(Reviews)
    #print(Weights)

    print("Generating Boosting Set")
    while len(NI) < max_size:
        for i in range(0, len(Weights)):
            temp = random.uniform(0, 1.0 / size)
            if (temp < Weights[i]):
                NI.append(i)
        
    random.shuffle(NI)
    
    print("BoostingSet Size: %d"%(len(NI)))
    for i in NI:
        NR.append(Reviews[i])
        NL.append(Labels[i])
    
    return NR, NL

def ExtendBoostingSet(Reviews, Lables, Weights,time):
    Weights = unify(Weights)
    
    NR = []
    NL = []
    NI = []
    size = len(Reviews)
    max_size = time * len(Reviews)

    print("Generating Boosting Set")
    for i in range(0, len(Reviews)):
        NI.append(i)
        
    while(len(NI) < max_size):
        for i in range(0, len(Weights)):
            temp = random.uniform(0, 1.0 / size)
            if (temp < Weights[i]):
                NI.append(i)

    random.shuffle(NI)

    print("BoostingSet Size: %d"%(len(NI)))
    for i in NI:
        NR.append(Reviews[i])
        NL.append(Lables[i])

    return NR,NL

def AdaBoost(X, Y, TX, algorithm='SVM', num=3, TreeDepth=1000, manner='Weighted', times=2):
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
        if (manner == 'EqualSample'):
            NX,NY = BoostingSet(X, Y, SW)
        if (manner == 'ExtendSample'):
            NX,NY = ExtendBoostingSet(X, Y, SW, times)
        #print(SW)
        #Train Weighted Classifier
        clf = None
        if (algorithm == 'SVM'):
            if (manner == 'Weighted'):
                print("Error! SVM can not use weighted method")
                return
            clf = SVM(NX, NY)
        if (algorithm == 'DTree'):
            if (manner == 'Weighted'):
                clf = DTree(X, Y, SW, TreeDepth)
            else:
                clf = DTree(NX, NY, None, TreeDepth)
        if (algorithm == 'KNN'):
            if (manner == 'Weighted'):
                print("Error! KNN can not use weighted method")
                return
            clf = KNN(NX,NY)

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
            print("Warning! A bad classifier arise. abort the loop")
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
    parser = argparse.ArgumentParser()

    parser.add_argument('DictionaryPath', help = 'The dictionary path to embedding')
    parser.add_argument('TrainPath', help = 'The Training Set path')
    parser.add_argument('TestPath', help = 'The Testing Set path')
    parser.add_argument('OutputPath', help = 'The Output path')
    parser.add_argument('-a', '--algorithm', default='DTree', choices=['SVM','DTree','KNN'], help = 'Using which kind of Algorithm to train(We support SVM, DTree, KNN and defualt is DTree)')
    parser.add_argument('-n', '--num' ,type=int, default=50, help = 'The num of weaker classifier\'s num, the default is 50')
    parser.add_argument('-e', '--ensemble', default='Bagging', choices=['Bagging','Boosting'], help = 'Using which kind of ensemble Algorithm to train(We support Bagging, Boosting and default is Bagging')
    parser.add_argument('-d', '--max_deep', type = int, default=100, help = 'The Max Deep for DTree in Boosting')
    parser.add_argument('-v', '--validation', help = 'Turn validation mode on', action = 'store_true')
    parser.add_argument('-b', '--boosting_method', default='Weighted',choices=['Weighted','EqualSample','ExtendSample'], help = 'boosting method including Weighted, EqualSample, ExtendSample NOTICE: SVM can not use Weighted method!')
    parser.add_argument('--extend_time', type = int, default=2, help = 'The Extended time for ExtendSample Method, default is 2 times')
    parser.add_argument('--validation_train_num', type = int, default=20000, help = '(Under validation mode) The train set size, default is 200000')
    parser.add_argument('--validation_test_num', type = int, default=1000, help = '(Undervalidation mode) The validation set size, default is 1000')

    args = parser.parse_args()
    
    dic_path = args.DictionaryPath
    train_path = args.TrainPath
    test_path = args.TestPath
    output_path = args.OutputPath

    algorithm = args.algorithm
    n = args.num
    ensemble = args.ensemble
    tree_depth = args.max_deep

    train_num = -1 #-1 means take all sample to train (including the validation set)
    val_num = 1000
    
    boosting_method = args.boosting_method
    extend_time = args.extend_time

    if args.validation:
        train_num = args.validation_train_num
        val_num = args.validation_test_num
    
    
    ts = 0     #train set start
    te = train_num    #train set end
    
    Dic = DG(dic_path)
    
    print("Load Train Data")
    X, Y = LoadData(train_path,Dic, ts, te)
    TX = []
    TY = []
    
    if args.validation:
        SP = te - val_num #Split Point
        assert(SP < te)
        TX = X[SP:-1]
        V_Y = Y[SP:-1]
        X = X[0:SP]
        Y = Y[0:SP]
    else:
        TX = LoadTestData(test_path, Dic)
        
    Ans = []
    
    if (ensemble == "Bagging"):
        Ans = Bagging(X, Y, TX, algorithm, n)
    if (ensemble == "Boosting"):
        Ans = AdaBoost(X, Y, TX, algorithm, n, tree_depth, boosting_method, extend_time)
    
    if args.validation:
        rate, rmse = validate(Ans, V_Y)
        print("Correct Rate: %s, RMSE: %s"%(str(rate), str(rmse)))
    else:
        Final = []
        tot = len(Ans)
        for i in range(1, tot+1):
            Final.append([i,Ans[i - 1]])
        pd.DataFrame(Final, columns = ['id','label']).to_csv(output_path)

