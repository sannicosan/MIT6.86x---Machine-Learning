##Libraries
import numpy as np


## Exercise 1-3: solved manually

## Exercise 4
def Ex4():
    V = ['A','B']
    W = ['AAx','ABB', 'BAA']
    W_encode = []

    for w in W:
        w_encode = []
        for c in w:
            if c == V[0]:
                w_encode.append([1,0])
            elif c ==  V[1]:
                w_encode.append([0, 1])
            else:
                w_encode.append([0, 0])

        W_encode.append(w_encode)
        W_encode_arr= np.array(W_encode)
    return  np.transpose(W_encode_arr, axes = (0,2,1))

## Exercise 5
def Ex5(W_encode):

    def RELU(z):
        return np.maximum(0, z)

    # Data


    T = W_encode.shape[0]                               # number of feature vectors per sentence
    V = W_encode.shape[1]                               # number of components ~ length(vocabulary)


    Wss = np.array([[-1,0],[0,1]])
    Wsx = np.eye(V)
    Wsy = Wsx.copy()
    W0 = 0

    ST = np.zeros((V, T), int).T


    for w,w_enc in enumerate(W_encode):
        st_prev = np.array([0, 0])
        for t,xt in enumerate(w_enc.T):
            if t == T:
                y = np.sign(Wsy @ st + W0)
            else:
                st = RELU(Wss @ st_prev + Wsx @ xt )
            # Update st_prev <- st
            st_prev = st.copy()

        ST[w,:] = st.reshape(1,-1)

    return ST.T

W_encode = Ex4()
print('Sentece encoded: \n', W_encode)


ST = Ex5(W_encode)
print('\n ST: \n', ST)
