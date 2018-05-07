import pandas as pd
import sys
import re
from tqdm import tqdm
dic = {}
if __name__ == "__main__":
    file_name = sys.argv[1]
    d = pd.read_csv(file_name)
    label = d['label']
    review = d['review']
    num = 0

    print("Generating Dictionary")
    for i in tqdm(review):
        l = i.split()
        print(l)
        for w in l:
            if w in dic:
                pass
            else:
                dic[w] = num
                num += 1
    
    print("Dictionary output")
    file_object = open("dic.txt",'w')
    for k in tqdm(dic.keys()):
        s = k + ':' + str(dic[k]) + '\n'
        file_object.write(s)
    file_object.close()
        
