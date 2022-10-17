import numpy as np
from scipy.stats import norm as Normal

#region E-Step

## Data
import utils

x = np.array([0.2,-0.9,-1,1.2,1.8]).reshape(5,1)

## Initializations
mu = np.array([-3,2]).reshape(1,2)
var = np.array([4,4]).reshape(1,2)
pj = np.array([0.5,0.5]).reshape(1,2)

## Posterior probability
N = utils.multi_normal(mu,var,x)
print('Normal probs:\n', N)

P = pj*N/np.sum(pj*N,axis = 1).reshape(-1,1)
print('Posterior probability:\n',P)

# N2 = Normal(mu,np.sqrt(var)).pdf(x)
# P2 = pj[0,0]*N2.T[0]/np.sum(pj*N2)
# print('Posterior probability:\n', P2)


#endregion