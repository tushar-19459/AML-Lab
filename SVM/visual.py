import matplotlib.pyplot as plt

def CreateScatter(X,y):
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
    plt.title("Linearly Separable Data")
    plt.show()


def CreateMargin(X,y,svm_head,x_val,y_decision,y_margin_neg,y_margin_pos,i):

    plt.figure(figsize=(7,6))
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k')

    plt.scatter(
        svm_head.support_vectors_[:,0],
        svm_head.support_vectors_[:,1],
        s=120,facecolors="none",edgecolors='k',linewidths=2,
        label="support vectors"
    )

    plt.plot(x_val,y_decision,'k-',label="decision Boundary")
    plt.plot(x_val,y_margin_pos,'k--',label="margin +1")
    plt.plot(x_val,y_margin_neg,'k--',label="margin -1")

    plt.fill_between(
        x_val,y_margin_pos,y_margin_neg,
        color='gray',alpha=0.2,label="margin area"
    )

    plt.xlabel("Feature 1")
    plt.xlabel("Fearure 2")
    plt.title(f"Soft Margin  SVM (C={i})")
    plt.legend()
    plt.show()