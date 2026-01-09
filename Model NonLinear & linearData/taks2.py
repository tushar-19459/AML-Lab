from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from data import gen_xor_data
# from visual import plot_2d_data

def main():
    X,y = gen_xor_data()

    # plot_2d_data(X,y,title="target")
    
    lr =LogisticRegression()
    lr.fit(X,y)
    y_pred = lr.predict(X)
    linear_acc = accuracy_score(y,y_pred)
    print("linear model accuracy on XOR data ",linear_acc)
    # plot_2d_data(X,y_pred,title="predicted")

    poly = PolynomialFeatures(degree=2,include_bias=False)
    poly_lr =LogisticRegression()
    
    X_poly = poly.fit_transform(X)
    lr.fit(X_poly,y)
    print(X_poly)
    y_pred = lr.predict(X_poly)
    linear_acc = accuracy_score(y,y_pred)
    print("poly linear model accuracy on XOR data ",linear_acc)
    # plot_2d_data(X,y_pred,title="predicted")


    



    
if __name__== "__main__":
    main()