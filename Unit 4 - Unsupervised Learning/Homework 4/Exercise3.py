import numpy as np

#region

## Data
import utils
from scipy.stats import norm as Normal

x = np.array([-1,0,4,5,6], dtype= np.float64).reshape(-1,1)
params = np.array([0.5,0.5,6,7,1,4]).reshape(1,-1)
pi = params[:,:2]
mu = params[:,2:4]
var = params[:, 4:]


N = Normal(mu,np.sqrt(var)).pdf(x)
# N = utils.multi_normal(mu,var,x)
print('Normal probs:\n', N)
P= np.sum(np.log(pi*N))

print('p_xi: ', P)




# print(N)

#endregion