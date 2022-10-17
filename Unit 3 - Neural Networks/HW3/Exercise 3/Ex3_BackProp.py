import numpy as np
from Exercise2.sigmoid import sigmoid as sigm

##region ################ Exercise 3: Back Propagation #####################
########################################################################

#region Last item

# Initialization
t = 1
x = 3
w1 = 0.01
w2 = -5
b = -1

# first part
z1 = w1*x
a1 = 0 if z1<0 else max(0,z1)
z2 = w2*a1 + b
y = sigm(z2)
C = 1/2*(y-t)**2

print('Loss = ', C)

# second part: derivatives
d_sigm = y**2 * np.exp(-z2)

dC_db = (y-1)*d_sigm
dC_dw2 = dC_db*a1
dC_dw1= dC_db*w2*x

print("dC/dw1 = ", dC_dw1)
print("dC/dw2 = ", dC_dw2)
print("dC/db = ", dC_db)


#endregion

