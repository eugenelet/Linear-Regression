import sys, getopt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np
from numpy.linalg import inv
from train_ML import *
from train_MAP import *
from test import *
from plot_diagram import *
from basis import *
from MSE import *
import csv

def main(argv):
	poly_gauss = False
	ML_MAP = False

	try:
		opts, args = getopt.getopt(argv,"hb:m:")
	except getopt.GetoptError:
		print 'bayesian.py -b <basis type> -m <method>'
		print '<basis type>: 0=polynomial 1=gaussian'
		print '<method>: 0=maximum likelihood 1=maximum a posteriori'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'bayesian.py -b <basis type> -m <method>'
			print '<basis type>: 0=polynomial 1=gaussian'
			print '<method>: 0=maximum likelihood 1=maximum a posteriori'
			sys.exit()
		elif opt in ("-b"):
			if arg == '1':
				poly_gauss = True
		elif opt in ("-m"):
			if arg == '1':
				ML_MAP = True




	##############################
	# HYPERPARAMETERS
	##############################

	if poly_gauss:
		# MAP
		global_basis_n = 38 
		global_var = 5
		global_lambda = 7
	else:
		# ML
		global_order = 10
		global_lambda = 2


	# Import Training Data
	xy_data = np.genfromtxt('X_train.csv', delimiter = ',')
	x = xy_data[:,0]
	y = xy_data[:,1]

	T_data = np.genfromtxt('T_train.csv')
	target = np.array(T_data)
	

	# x_sorted, y_sorted = zip(*sorted(zip(x, y)))
	# x_sorted, t_sorted = zip(*sorted(zip(x, T_data)))


	# plot_contour(x,y,T_data,3)
	# my_plot_surface(x,y,T_data,4)


	##############################
	# CROSS VALIDATION
	##############################

	# GAUSSIAN

	train_num = len(xy_data) * 0.6
	cross_num = len(xy_data) * 0.4
	
	train_x = xy_data[0:int(train_num),0]
	train_y = xy_data[0:int(train_num),1]
	train_target = np.array(T_data[0:int(train_num)])
	
	cross_x = xy_data[int(train_num)+1:int(train_num)+int(cross_num),0]
	cross_y = xy_data[int(train_num)+1:int(train_num)+int(cross_num),1]
	cross_target = np.array(T_data[int(train_num)+1:int(train_num)+int(cross_num)])
	
	global_error = 100000000
	for basis_n in range(30,45):
		for var in range(3,8):
			for _lambda in range(1,10):
				basis = joint_gaussian_basis_2d(train_x,train_y,basis_n,var*50)
				weight_MAP = train_MAP(basis, train_target, _lambda*0.1)
				basis = joint_gaussian_basis_2d(cross_x,cross_y,basis_n,var*50)
				res = test(basis, weight_MAP)
				error = MSE(res,cross_target)
				print 'ERROR: %f basis_n: %d var: %d lambda: %d' %(error,basis_n,var,_lambda)
				if(error < global_error):
					global_error = error
					global_basis_n = basis_n
					global_var = var
					global_lambda = _lambda
	
	print '=================================='
	print 'RESULT:   ERROR: %d basis: %d var: %d lambda: %d' %(global_error,global_basis_n,global_var,global_lambda)
	print '=================================='




	# POLYNOMIAL
	#train_num = len(xy_data) * 0.6
	#cross_num = len(xy_data) * 0.4
	#
	#train_x = xy_data[0:int(train_num),0]
	#train_y = xy_data[0:int(train_num),1]
	#train_target = np.array(T_data[0:int(train_num)])
	#
	#cross_x = xy_data[int(train_num)+1:int(train_num)+int(cross_num),0]
	#cross_y = xy_data[int(train_num)+1:int(train_num)+int(cross_num),1]
	#cross_target = np.array(T_data[int(train_num)+1:int(train_num)+int(cross_num)])
	#
	#global_error = 100000000
	#for order in range(3,20):
	#		basis = polynomial_basis(train_x,train_y,order)
	#		weight_ML = train_ML(basis, train_target);
	#		basis = polynomial_basis(cross_x,cross_y,order)
	#		res = test(basis, weight_ML)
	#		error = MSE(res,cross_target)
	#		print 'ERROR: %d order: %d' %(error,order)
	#		if(error < global_error):
	#			global_error = error
	#			global_order = order
	#
	#
	#print '=================================='
	#print 'RESULT:   ERROR: %d order: %d' %(global_error,global_order)
	#print '=================================='




	# GAUSSIAN
	#train_num = len(xy_data) * 0.6
	#cross_num = len(xy_data) * 0.4
	#
	#train_x = xy_data[0:int(train_num),0]
	#train_y = xy_data[0:int(train_num),1]
	#train_target = np.array(T_data[0:int(train_num)])
	#
	#cross_x = xy_data[int(train_num)+1:int(train_num)+int(cross_num),0]
	#cross_y = xy_data[int(train_num)+1:int(train_num)+int(cross_num),1]
	#cross_target = np.array(T_data[int(train_num)+1:int(train_num)+int(cross_num)])
	#
	#global_error = 100000000
	#for basis_n in range(34,42):
	#	for var in range(2,7):
	#		basis = gaussian_basis_2d(train_x,train_y,basis_n,var*50)
	#		weight_ML = train_ML(basis, train_target);
	#		basis = gaussian_basis_2d(cross_x,cross_y,basis_n,var*50)
	#		res = test(basis, weight_ML)
	#		error = MSE(res,cross_target)
	#		print 'ERROR: %d basis: %d var: %d' %(error,basis_n,var)
	#		if(error < global_error):
	#			global_error = error
	#			global_basis_n = basis_n
	#			global_var = var
	#
	#
	#print '=================================='
	#print 'RESULT:   ERROR: %d basis: %d var: %d' %(global_error,global_basis_n,global_var)
	#print '=================================='






	##############################
	# TRAIN
	##############################

	# Basis #
	if poly_gauss:
		basis = joint_gaussian_basis_2d(x,y,global_basis_n,global_var*50)
	else:
		basis = polynomial_basis(x,y,global_order)

	# Method #
	if ML_MAP:
		weight_MAP = train_MAP(basis, target, global_lambda*0.1)
	else:
		weight_ML = train_ML(basis, target)




	##############################
	# TEST 
	##############################

	# test_data = np.genfromtxt('X_test.csv', delimiter = ',')
	# x = test_data[:,0]
	# y = test_data[:,1]

	test_data = np.genfromtxt('X_train.csv', delimiter = ',')
	x = test_data[0:10000,0]
	y = test_data[0:10000,1]

	test_result = np.genfromtxt('T_train.csv')
	t = test_result[0:10000]

	# Basis #
	if poly_gauss:
		basis = joint_gaussian_basis_2d(x,y,global_basis_n,global_var*50)
	else:
		basis = polynomial_basis(x,y,global_order)

	# Method #
	if ML_MAP:
		z = test(basis, weight_MAP)
	else:
		z = test(basis, weight_ML)


	print z.shape
	print t.shape

	print MSE(z,t)

	###########################
	# PLOT
	###########################

	plot_contour(x,y,z,1)
	my_plot_surface(x,y,z,2)

	if poly_gauss:
		if ML_MAP:
			np.savetxt('MAP.csv', z, fmt='%i', delimiter='\n') 
			np.savetxt('Bayesian.csv', z, fmt='%i', delimiter='\n') 
		else:
			np.savetxt('ML.csv', z, fmt='%i', delimiter='\n') 
	else:
		print 'Please use Gaussian Basis'


	plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])