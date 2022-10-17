import numpy as np

def simple_nn(x, W, V, f_hid, f_out):

    z = W[:,:-1]@ x.T + W[:,-1]

    f_z = f_hid(z)

    u = V[:,:-1] @ f_z + V[:,-1]

    return f_out(f_hid(u))


