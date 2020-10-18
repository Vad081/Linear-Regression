import numpy as np
import matplotlib.pyplot as plt
from grad import compute_cost, gradient_descent

data = np.loadtxt('ex1data1.txt',delimiter = ',')

m = len(data[:,1])

X = np.append(np.ones((m,1)),data[:,0].reshape(-1,1),axis=1)
y = data[:,1].reshape(-1,1)
theta = np.zeros((2,1))


if __name__== '__main__':
    print(compute_cost(X, y, theta))
    theta,history = gradient_descent(X,y,theta,0.01,2000)
    x_values = [x for x in range(0,25)]
    y_values = [y*theta[1]+theta[0] for y in x_values]
    plt.scatter(data[:,0],data[:,1])
    plt.plot(x_values,y_values,c='r')
    plt.show()




