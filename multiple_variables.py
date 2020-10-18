import numpy as np
import pandas as pd
from grad import compute_cost, gradient_descent, normalization

data2 = np.loadtxt('ex1data2.txt',delimiter=',')

m = len(data2[:,-1])
X = data2[:,:2].reshape(-2,2)
X2, mean, std = normalization(X)
X2 = np.append(np.ones((m,1)),X2,axis=1)
y = data2[:,-1].reshape(-1,1)
theta = np.zeros((3,1))

if __name__ == '__main__':
    print(compute_cost(X2,y,theta))
    theta, history = gradient_descent(X2, y, theta, 0.01, 400)
    print(theta)
    
