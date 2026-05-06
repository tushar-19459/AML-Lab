# code for overlapping data 
from data import OverLappingData
from sklearn.svm import SVC
from visual import CreateMargin,CreateScatter
import numpy as np

#gen data
X,y = OverLappingData()

#create svm obj softmargin 
C_values = [1,10,100,1000,1000]
for i in C_values:
    svm_head = SVC(kernel='linear',C=i)
    svm_head.fit(X,y)

    w = svm_head.coef_[0]
    b = svm_head.intercept_[0]

    x_val = np.linspace(X[:,0].min()-1,X[:,0].max()-1,200)
    y_decision = -(w[0]*x_val+b)/w[1]

    y_margin_pos = -(w[0] * x_val+b-1)/w[1]
    y_margin_neg = -(w[0] * x_val+b+1)/w[1]

    print(" W ",w)
    print(" b ",b)
    print("number of support vectors",len(svm_head.support_vectors_))

    # CreateScatter(X,y)
    CreateMargin(X,y,svm_head,x_val,y_decision,y_margin_neg,y_margin_pos,i)