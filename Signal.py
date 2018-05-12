from tqdm import tqdm
from feature import RD

import pandas as pd
from embedding import DG
from embedding import RE
from sklearn import tree
from sklearn import svm
import sys
import math

def LoadData(file_name,dic,start,end): #Load Traing Data
    #assert(start <= end)
    Labels = []
    Reviews = []
    d = pd.read_csv(file_name)[start:end]
    label = list(d['label'])
    reviews = list(d['review'])
    print("Loading Label..")
    for i in tqdm(label):
        Labels.append(int(i))
    
    print("Loading Review..")
    Reviews = RE(reviews,dic)
    
    #Reviews = RD(Reviews, Labels, 2000)

    return Reviews,Labels

def LoadTestData(file_name,dic): #Load Test Data
    Labels = []
    Reviews = []
    d = pd.read_csv(file_name)
    reviews = d['review']
    print("Loading Testing Review..")
    Reviews = RE(reviews,dic)
    
    #Reviews = RD(Reviews, Labels, 2000)
    
    return Reviews

def DTree(Reviews, Labels, sw=None, Deep=None): #Decision Tree
    clf = tree.DecisionTreeClassifier(max_depth=Deep)
    
    print("Start DTree training...")
    clf = clf.fit(Reviews,Labels,sample_weight=sw)
    print("End DTree training...")

    print("DTree depth: %d" % clf.n_features_)

    return clf


def SVM(Reviews, Labels, sw=None):
    lin_clf = svm.LinearSVC()
    
    print("Start SVM training...")
    lin_clf.fit(Reviews, Labels, sample_weight=sw)
    print("End SVM training...")

    return lin_clf

def SVC(Reviews, Labels, sw=None):
    clf = svm.SVC()
    
    print("Start SVM training...")
    clf.fit(Reviews, Labels, sample_weight=sw)
    print("End SVM training...")

    return clf


def validate(Ans, Real):
    correct = 0.0
    total = len(Real)
    RMSE = 0.0
    print("Validate")
    for i in tqdm(range(0, total)):
        if (Ans[i] == Real[i]):
            correct += 1
        RMSE += (Real[i] - Ans[i]) * (Real[i] - Ans[i])
    RMSE = math.sqrt(RMSE / total)
    return (correct / total), RMSE

if __name__ == "__main__":
    dic_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    algorithm = sys.argv[4]
    #n = int(sys.argv[5])
    #ensemble = sys.argv[6]
    mode = sys.argv[5]
    if (mode == 'test'):
        output_file = sys.argv[6]
    train_num = -1 #-1 means take all sample to train (including the validation set)
    val_num = 500
    if (mode == 'validation'):
        train_num = int(sys.argv[6])
        val_num = int(sys.argv[7])

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
    if (algorithm == "SVM"):
        Ans = SVM(X, Y).predict(TX)
    if (algorithm == "DTree"):
        Ans = DTree(X, Y).predict(TX)

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
        
