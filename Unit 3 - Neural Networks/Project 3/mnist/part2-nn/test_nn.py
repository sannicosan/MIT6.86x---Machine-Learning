import numpy as np
import neural_nets as nn


def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def check_rectified_linear_unit(): 
    try:
        x = 1.
        nn.rectified_linear_unit(x)
    except NotImplementedError:
        print(yellow("FAIL"), ": ReLU not implemented!")
        return 
    # Test for different values.
    x = np.array([-5., 0., 5])
    for i in range(len(x)):
        if not nn.rectified_linear_unit(x[i]) == np.maximum(0,x[i]):
            print(red("FAIL"), ": ReLU gives wrong output!")
            return
        
    print(green("PASS"), ": ReLU Implemented.")
    
    
def check_rectified_linear_unit_derivative(): 
    try:
        x = 1.
        nn.rectified_linear_unit_derivative(x)
    except NotImplementedError:
        print(yellow("FAIL"), ": ReLU derivative not implemented!")
    
    x = np.array([-5., 0., 5])
    y = np.array([0., 0., 1. ])
    for i in range(len(x)):
        if not nn.rectified_linear_unit_derivative(x[i]) == y[i]:
            print(red("FAIL"), ": ReLU derivative gives wrong output!")
            return
        
    print(green("PASS"), ": ReLU derivative Implemented.")
    
    
def main():
    check_rectified_linear_unit()
    check_rectified_linear_unit_derivative()
    
if __name__ == "__main__":
    main()
