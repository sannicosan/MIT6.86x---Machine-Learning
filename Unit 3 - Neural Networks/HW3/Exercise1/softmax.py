import numpy as np

def softmax(x):

    e_x = np.exp(x)

    return e_x/np.sum(e_x)

def softmax_beta(x,B):

    e_Bx = np.exp(B*x)

    return e_Bx/np.sum(e_Bx)