import numpy as np
from numpy.linalg import inv

# Obtain variance term of bayesian
def bayesian_var(basis, _lambda):
	s_n_1 = _lambda*np.identity(basis.shape[0]) + np.dot(basis,basis.T)
	var = 1 + np.dot(basis.T, np.dot(inv(s_n_1),basis))
	return var
