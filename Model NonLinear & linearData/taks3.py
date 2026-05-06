from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from data import gen_xor_data
from visual import plot_2d_data

def main():
   X,y = gen_xor_data()
   rbf_svm = SVC(kernel='rbf',gamma='scale')
   rbf_svm.fit(X,y)
   y_pred_rbf = rbf_svm.predict(X)
   plot_2d_data(X,y_pred_rbf,title="RBF kernal SVM Prediction")

   print("linear SVM Accuracy ",accuracy_score(y,y_pred_rbf))

    
if __name__== "__main__":
    main()