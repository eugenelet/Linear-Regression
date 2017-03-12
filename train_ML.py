import numpy as np
from numpy.linalg import inv

def train_ML(basis, target):
	return np.dot(np.dot(inv(np.dot(basis,basis.T)),basis),target.T);

