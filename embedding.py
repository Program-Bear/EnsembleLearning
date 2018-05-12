import pandas as pd
import sys
import numpy as np
from tqdm import tqdm



def DG(path): #Dictionary Generating
    print("build the dictionary")
    dic = {}
    file_object = open(path,'r')
    for line in tqdm(file_object.readlines()):
        line = line.strip()
        key = line.split(':')[0]
        value = line.split(':')[1]
        dic[key] = value
    file_object.close()
    return dic

def DO(r_dic,path):
    file_object = open(path,'w')
    for key in r_dic.keys():
        value = r_dic[key]
        s = str(key) + ":" + str(value) + '\n'
        file_object.write(s)
    file_object.close()

def RE(review,dic): #Review Embedding
    dim = len(dic.keys())
    total_num = len(review)

    print("dim: %d"%dim)
    
    Embeddings = []
    print("Calculate embedding")
    for i in tqdm(range(0,total_num)):
        embedding = [0 for j in range(0,dim)]
        r = review[i].split()
        for w in r:
            try:
                n = int(dic[w])
            except:
                continue
            embedding[n] += 1
        Embeddings.append(embedding)
    
    return Embeddings


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


if __name__ == "__main__":
    input_name = sys.argv[1]
    dic_name = 'dic.txt'
#    output_name = sys.argv[2]
    
    
    ans = []
    d = pd.read_csv(input_name)
    dic = DG(dic_name)
    #label = d['label']
    review = d['review']    
    print(RE(review,dic)[0])

#    print("Output embedding")
#    file_object = open(output_name, 'w')
#    for i in tqdm(ans):
#        file_object.write(i)
#    file_object.close()
        
