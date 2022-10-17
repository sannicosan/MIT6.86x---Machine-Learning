##Libraries
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt


#region constants
MAX_ITER = 10000
PI = np.pi

## Perceptron algorithm
def perceptron(data,th,T, offset = True):

    n_updates = 0
    while (1):

        x = data[0]
        y = data[1]
        n = len(y)
        n_errors = 0

        for i in range(n):
            if y[i] * (np.dot(th[1:].T, x[:, i]) + th[0]) <= 0:
                th[1:] = th[1:] + y[i] * x[:, i].reshape(-1,1)
                if offset: th[0] = th[0] + y[i]
                n_errors += 1

        n_updates += n_errors

        if (not n_errors or n_updates == T):
            break

    return th,n_updates

#region Data
## Data
x = np.array([[1,1],[2,3],[3,4],[-.5,-.5],[-1,-2],[-2,-3],[-3,-4],[-4,-3]]).T
y = np.array([1,1,1,-1,-1,-1,-1,-1]).reshape(-1,1)
data = (x,y)
n= len(y)
##endregion

## Exercise 1
def Ex1(data):
    ## Initialization
    th = np.array([0,0,0]).reshape(-1,1)
    ## perceptron
    th,n_upd = perceptron(data,th,MAX_ITER)
    print('Perceptron performed {} updates'.format(n_upd))
    print('Parameters: ', th)

    ## Plotting
    colors = ['r','g'] 
    plt.scatter( x = x[0,:], y = x[1,:], c = y, cmap = matplotlib.colors.ListedColormap(colors))

    x_lin = np.linspace(-5,4)
    y_lin = - th[1]/th[2]*x_lin - th[0]/th[2]
    plt.plot(x_lin,y_lin, 'k-')

    plt.show()

## Exercise 2
def Ex2(data):
    ## Initialization
    th = np.array([0,1,1]).reshape(-1,1)
    ## perceptron
    th,n_upd = perceptron(data,th,MAX_ITER,False)
    print('Perceptron performed {} updates'.format(n_upd))
    print('Parameters: ', th)

    ## Plotting
    colors = ['r','g'] 
    plt.scatter( x = x[0,:], y = x[1,:], c = y, cmap = matplotlib.colors.ListedColormap(colors))

    x_lin = np.linspace(-5,4)
    y_lin = - th[1]/th[2]*x_lin - th[0]/th[2]
    plt.plot(x_lin,y_lin, 'k-')

    plt.show()

## Exercise 3
def Ex3(data):
    ## Initialization
    R = np.array([[np.cos(PI/3),-np.sin(PI/3)],[np.sin(PI/3),np.cos(PI/3)]])

    x,y = data
    x2_rot =  R @ x[:,1]
    print('(2,3) rotated:', x2_rot)


## Exercise 4
def Ex4(data):
    ## Initialization
    th = np.array([0, 0, 0]).reshape(-1, 1)
    R = np.array([[np.cos(PI/3),-np.sin(PI/3)],[np.sin(PI/3),np.cos(PI/3)]])
    x,y = data

    x_rot = np.zeros(x.shape)
    for i in range(n):
        x_rot[:,i] = R @ x[:,i]

    data_rot = (x_rot,y)
    # ## perceptron
    th, n_upd = perceptron(data_rot, th, MAX_ITER)
    print('Perceptron performed {} updates'.format(n_upd))
    print('Parameters: ', th)
    #
    # ## Plotting
    colors = ['r','g'] 
    plt.scatter(x=x_rot[0, :], y=x_rot[1, :], c=y, cmap=matplotlib.colors.ListedColormap(colors))

    x_lin = np.linspace(-5, 4)
    y_lin = - th[1] / th[2] * x_lin - th[0] / th[2]
    plt.plot(x_lin, y_lin, 'k-')

    plt.show()

# Ex1(data)
# Ex2(data)
# Ex3(data)
# Ex4(data)


