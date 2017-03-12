import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from basis import *
from train_MAP import *
from train_Bayesian import *
from test import *
from MSE import *
resolution = 500

gauss = np.zeros((resolution,))
x = np.zeros((resolution,))
for i in range(1,resolution):
	gauss[i] = np.exp(-0.5*(i - 75)**2/500) + np.exp(-0.5*(i - 311)**2/360) + np.exp(-0.5*(i - 210)**2/800)
	x[i] = i
	# print '%d : %f' % (i, gauss[i])

plt.figure(1)
plt.plot(gauss)




# GAUSSIAN

global_error = 100000000
for basis_n in range(2,10):
	for var in range(1,10):
		for _lambda in range(1,10):
			basis = gaussian_basis(x, basis_n, var*50)
			train_weight = train_MAP(basis,gauss,_lambda*0.1) 
			res = test(basis,train_weight)
			error = MSE(res,gauss)
			print 'ERROR: %f basis: %d var: %d lambda: %d' %(error,basis_n,var,_lambda)
			if(error < global_error):
				global_error = error
				global_basis_n = basis_n
				global_var = var
				global_lambda = _lambda


print '=================================='
print 'RESULT:   ERROR: %f basis: %d var: %d lambda: %d'  %(global_error,global_basis_n,global_var,global_lambda)
print '=================================='



basis = gaussian_basis(x, global_basis_n, global_var*50)
train_weight = train_MAP(basis,gauss,global_lambda) 

var = bayesian_var(basis, global_lambda)
test_res = np.dot(basis.T,train_weight)
# print test_res

plt.figure(2)
x_axis = np.linspace(min(x),max(x))
plt.plot(x,test_res, 'k-')
var_mean = np.mean(var,axis=0)
std_dev = var_mean**0.5
plt.fill_between(x,test_res-std_dev/2,test_res+std_dev/2)
plt.figure(3)
plt.plot(x,test_res, 'k-')
for i in range(0,basis.shape[0]):
	plt.plot(x,basis[i])


plt.show()

