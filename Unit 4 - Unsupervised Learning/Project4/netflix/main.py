import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# #region Ex2: Kmeans
# K = [1,2,3,4]
# seed = [0,1,2,3,4]
# min_cost = np.inf
# for k in K:
#     for s in seed:
#     ## E-step
#         gauss_mix, _ = common.init(X,k,s)
#         post_p = kmeans.estep(X,gauss_mix)
#         ## M-Step
#         gauss_mix, post_p,current_cost = kmeans.run(X,gauss_mix,post_p)
#         if (current_cost < min_cost):
#             min_cost = current_cost
#             min_seed = s
#             min_gauss_mix = gauss_mix
#             min_post_p = post_p
#
#     print('(K = '+ str(k)+')\tSeed: {}; Cost: {}'.format(min_seed,min_cost))
#     common.plot(X,min_gauss_mix,min_post_p, 'K = ' + str(k))
#
# #endregion

#region Ex3 - EM Collaborative filtering
K = 3
gauss_mix,post =  common.init(X,K,0)
post_p,LL = naive_em.estep(X,gauss_mix)
print('Posterior: \n', post_p)
print('\n  Log-Likelihood:', LL)
#endregion
