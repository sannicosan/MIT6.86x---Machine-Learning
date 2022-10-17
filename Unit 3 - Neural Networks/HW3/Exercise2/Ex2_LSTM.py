
import numpy as np
from Exercise2.LSTM import lstm

##region ################ Exercise 2: LSTM #####################
########################################################################




#endregion

# Hyper-parameters
Wfh = 0; Wfx = 0 ;bf = -100; Wch = -100
Wih = 0; Wix = 100; bi = 100; Wcx = 50
Woh = 0; Wox = 100; bo = 0; bc = 0

#region Item 1: LSTM States


# Input
x = np.array([0,0,1,1,1,0]).T

(c,h) = lstm(x,Wfh,Wfx,bf, Wch ,Wih , Wix, bi , Wcx ,Woh , Wox , bo , bc)
print('', h)
#endregion

#region Item 2:LSTM States 2
x = np.array([1,1,0,1,1]).T
(c,h) = lstm(x,Wfh,Wfx,bf, Wch ,Wih , Wix, bi , Wcx ,Woh , Wox , bo , bc)
print('', h)
#endregion

