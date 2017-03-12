import numpy as np
from numpy.linalg import inv

def train_MAP(basis, target, lamda):
	return np.dot(np.dot(inv(lamda*np.identity(basis.shape[0]) + np.dot(basis,basis.T)),basis),target.T);

