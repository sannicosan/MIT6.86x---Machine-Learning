import numpy as np
def sigmoid(x):
    if not isinstance(x,np.vectorize):
        return 1/(1+np.exp(-x))


    return  np.vectorize(lambda x: 1/(1+np.exp(-x)))