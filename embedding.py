import pandas as pd
import sys
import numpy as np
from tqdm import tqdm

def DG(path): #Dictionary Generating
    print("build the dictionary")
    dic = {}
    file_object = open(path,'r')
    for line in tqdm(file_object.readlines()):
        key = line.split(':')[0]
        value = line.split(':')[1]
        dic[key] = value
    file_object.close()
    return dic

def RE(review,dic): #Review Embedding
    dim = len(dic.keys())
    total_num = len(review)

    print("dim: %d"%dim)
    
    Embeddings = []
    z = [0 for i in range(0,dim)]
    print("Calculate embedding")
    for i in tqdm(range(0,total_num)):
        embedding = [0 for i in range(0,dim)]
        r = review[i].split()
        for w in r:
            try:
                n = int(dic[w])
            except:
                continue
            embedding[n] += 1
        Embeddings.append(embedding)
    
    return Embeddings

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
        
