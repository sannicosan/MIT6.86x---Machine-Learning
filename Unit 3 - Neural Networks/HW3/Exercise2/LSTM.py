from Exercise2 import *
from Exercise2.sigmoid import sigmoid as sigm
import numpy as np


def lstm(x,Wfh,Wfx,bf, Wch ,Wih , Wix, bi , Wcx ,Woh , Wox , bo , bc):
    # Hidden Gates
    h = []
    c = []
    for t,xt in enumerate(x):

        ht_1 = 0 if t == 0 else h[t - 1]
        ct_1 = 0 if t == 0 else c[t - 1]

        ft = sigm(Wfh * ht_1 + Wfx * xt + bf)   ## Forget Fate
        it = sigm(Wih * ht_1 + Wix * xt + bi)   ## input gate
        ot = sigm(Woh * ht_1 + Wox * xt + bo)   ## Output Gate

        # Memory cell
        ct = ft*ct_1 + it*np.tanh(Wch * ht_1 + Wcx * xt + bc)
        c.append(ct)
        # Visible Gate
        ht = round(ot * np.tanh(ct))
        h.append(ht)

    return(c,h)

