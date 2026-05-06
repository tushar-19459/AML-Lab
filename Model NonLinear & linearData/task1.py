from data import gen_xor_data
from visual import plot_2d_data

def main():
    X,y = gen_xor_data()
    plot_2d_data(X,y,title="Non-Linearly Separable Data")
    
if __name__== "__main__":
    main()