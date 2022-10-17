# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

##region ################ Exercise 1: Simple Neural Network #####################
########################################################################




#endregion


# Hyper-parameters
W = np.array([[1,0,-1],[0,1,-1],[-1,0,-1],[0,-1,-1]])
V = np.array([[1,1,1,1,0],[-1,-1,-1,-1,2]])

# #region Item 1: Feed Forward Step
# # Input
# x = np.array([3,14])
#
# # Activation functions
# fh = lambda y: np.maximum(y,0)      ## Hidden layer
# fout = lambda y: sfmx.softmax(y)    ## Output Layer
#
# y_corrector = snn.simple_nn(x,W,V,fh,(lambda y: y))
# y = snn.simple_nn(x,W,V,fh,fout)
#
# print('Simple NN Output (without softmax): ',y_corrector)
# print('Simple NN Output: ',y)
#
# #endregion

# #region Item 2: Decision Boundaries
#
# for i,wi in enumerate(W):
#
#     if wi[0] == 0:
#         a = - wi[0] / wi[1]
#         xx = np.linspace(-1, 1)
#         yy = a * xx - wi[2] / wi[1]
#         plt.plot(xx, yy, label="Line " + str(i + 1))
#     else:
#         a = - wi[2] / wi[0]
#         yy = np.linspace(-1, 1)
#         plt.plot(np.tile(a,len(yy)), yy, label="Line " + str(i + 1))
#
#
# plt.legend()
# plt.show()
#
#
#
# #endregion:

# #region Item 3: Output NN
#
# u_1 = np.array([1,-1]).T + V[:,-1]
# u_2 = V[:,-1]
# u_3 = 3*np.array([1,-1]).T + V[:,-1]
# fh = lambda y: np.maximum(y,0)      ## Hidden layer
#
# o_1 = sfmx.softmax(fh(u_1))
# o_2 = sfmx.softmax(fh(u_2))
# o_3 = sfmx.softmax(fh(u_3))
#
#
# print("Output 1 (o1): ", o_1[0])
# print("Output 2 (o1): ", o_2[0])
# print("Output 3 (o1): ", o_3[0])
#
# #endregion

# #region Item 4: Inverse Temperature
#
# beta = [1,3]
#
# print("1. f1 - f2 <= ", np.log(10^3-1)/beta[0])
# print("2. f1 - f2 <= ", np.log(10^3-1)/beta[1])
#
# #endregion
    