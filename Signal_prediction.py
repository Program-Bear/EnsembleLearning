from tqdm import tqdm
import pandas as pd
from embedding import DG
from embedding import RE
from sklearn import tree
from sklearn import svm
import sys

def LoadData(file_name,dic): #Load Traing Data
    Labels = []
    Reviews = []
    d = pd.read_csv(file_name)
    label = d['label']
    reviews = d['review']
    print("Loading Training Label..")
    for i in tqdm(label):
        Labels.append(int(i))
    
    print("Loading Training Review..")
    Reviews = RE(reviews,dic)
    
    return Reviews,Labels

def LoadTestData(file_name,dic): #Load Test Data
    Labels = []
    Reviews = []
    d = pd.read_csv(file_name)
    reviews = d['review']
    print("Loading Testing Review..")
    Reviews = RE(reviews,dic)
    
    return Reviews

def DecisionTree(Reviews, Labels):
    pass

def SVM(Reviews, Labels):
    pass

if __name__ == "__main__":
    dic_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    algorithm = sys.argv[4]

    Dic = DG(dic_path)
    X, Y = LoadData(train_path,Dic)
    T_X = LoadTestData(test_path,Dic)
    #print(X)
    #print(Y)
    #print("...")
    #print(X[0])
    #print("...")
    #print(T_X[0])
    
