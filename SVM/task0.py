# code of saparabel data 
from data import LinearSeparableData
from sklearn.svm import SVC
from visual import CreateMargin,CreateScatter
import numpy as np

#gen data
X,y = LinearSeparableData()

#create svm obj 
svm_head = SVC(kernel='linear',C=1e6)
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
CreateMargin(X,y,svm_head,x_val,y_decision,y_margin_neg,y_margin_pos,1e6)