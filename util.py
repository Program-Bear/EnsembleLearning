import math
def unify(S):
    NS = []
    num = len(S)
    tot = float(sum(S))
    for i in range(0, num):
        NS.append(S[i] / tot)
    #print(NS)
    return NS
