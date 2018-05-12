from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel

def RD(Raw, Label, Dim):
    raw_dim = len(Raw[0])
    
    print("Start Dimension Reduction")
    Reviews = SelectKBest(mutual_info_classif, k=Dim).fit_transform(Raw, Label)
    print("Dimension Reduction from %d to %d using chi2"%(raw_dim, Dim))
    
    return Reviews

if __name__ == "__main__":
    pass
