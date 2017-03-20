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
		# global_basis_n = 38 
		global_basis_n = 85 
		global_var = 9
		global_lambda = 7
	else:
		global_order = 10
		global_lambda = 2


	# Import Training Data
	xy_data = np.genfromtxt('X_train.csv', delimiter = ',')
	x = xy_data[:,0]
	y = xy_data[:,1]

	T_data = np.genfromtxt('T_train.csv')
	target = np.array(T_data)

	# plot_contour(x,y,T_data,3)
	# my_plot_surface(x,y,T_data,4)


	##############################
	# CROSS VALIDATION
	##############################

	# GAUSSIAN

	#train_num = len(xy_data) * 0.8
	#cross_num = len(xy_data) * 0.2
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
	#for var in range(3,8):
	#	for _lambda in range(1,10):
	#		basis = joint_gaussian_basis_2d(train_x,train_y,global_basis_n,(var*5)**2)
	#		weight_MAP = train_MAP(basis, train_target, _lambda*0.1)
	#		basis = joint_gaussian_basis_2d(cross_x,cross_y,global_basis_n,(var*5)**2)
	#		res = test(basis, weight_MAP)
	#		error = MSE(res,cross_target)
	#		print 'ERROR: %d basis_n: %d var: %d lambda: %d' %(error,global_basis_n,var,_lambda)
	#		if(error < global_error):
	#			global_error = error
	#			global_basis_n = global_basis_n
	#			global_var = var
	#			global_lambda = _lambda
	#
	#print '=================================='
	#print 'RESULT:   ERROR: %d basis: %d var: %d lambda: %d' %(global_error,global_basis_n,global_var,global_lambda)
	#print '=================================='



	"""
	train_num = len(xy_data) * 0.8
	cross_num = len(xy_data) * 0.2
	
	train_x = xy_data[0:int(train_num),0]
	train_y = xy_data[0:int(train_num),1]
	train_target = np.array(T_data[0:int(train_num)])
	
	cross_x = xy_data[int(train_num)+1:int(train_num)+int(cross_num),0]
	cross_y = xy_data[int(train_num)+1:int(train_num)+int(cross_num),1]
	cross_target = np.array(T_data[int(train_num)+1:int(train_num)+int(cross_num)])
	
	global_error = 100000000
	for var in range(9,14):
		basis = joint_gaussian_basis_2d(train_x,train_y,global_basis_n,(var*4)**2)
		weight_ML = train_ML(basis, train_target)
		basis = joint_gaussian_basis_2d(cross_x,cross_y,global_basis_n,(var*4)**2)
		res = test(basis, weight_ML)
		error = MSE(res,cross_target)
		print 'ERROR: %d basis_n: %d var: %d ' %(error,global_basis_n,var)
		if(error < global_error):
			global_error = error
			global_basis_n = global_basis_n
			global_var = var

	print '=================================='
	print 'RESULT:   ERROR: %d basis: %d var: %d' %(global_error,global_basis_n,global_var)
	print '=================================='

	"""



	train_num = len(xy_data) * 0.3
	cross_num = len(xy_data) * 0.7
	
	train_x = xy_data[0:int(train_num),0]
	train_y = xy_data[0:int(train_num),1]
	train_target = np.array(T_data[0:int(train_num)])
	
	cross_x = xy_data[int(train_num)+1:int(train_num)+int(cross_num),0]
	cross_y = xy_data[int(train_num)+1:int(train_num)+int(cross_num),1]
	cross_target = np.array(T_data[int(train_num)+1:int(train_num)+int(cross_num)])
	
	global_error = 100000000
	for _lambda in range(7,8):
		basis = joint_gaussian_basis_2d(train_x,train_y,global_basis_n,((1081/2)/global_basis_n)**2)
		weight_ML = train_ML(basis, train_target)
		basis = joint_gaussian_basis_2d(cross_x,cross_y,global_basis_n,((1081/2)/global_basis_n)**2)
		res = test(basis, weight_ML)
		error = MSE(res,cross_target)
		print 'ERROR: %d lambda: %d' %(error,_lambda)
		if(error < global_error):
			global_error = error
			global_lambda = _lambda

	print '=================================='
	print 'RESULT:   ERROR: %d lambda: %d' %(global_error,global_lambda)
	print '=================================='



	##############################
	# TEST 
	##############################

	test_data = np.genfromtxt('X_test.csv', delimiter = ',')
	x = test_data[:,0]
	y = test_data[:,1]

	# test_data = np.genfromtxt('X_Test.csv', delimiter = ',')
	# x = test_data[0:10000,0]
	# y = test_data[0:10000,1]

	# test_result = np.genfromtxt('T_train.csv')
	# t = test_result[0:10000]

	# Basis #
	if poly_gauss:
		# basis = joint_gaussian_basis_2d(x,y,global_basis_n,(global_var*4)**2)
		basis = joint_gaussian_basis_2d(x,y,global_basis_n,((1081/2)/global_basis_n)**2)
	else:
		basis = polynomial_basis(x,y,global_order)

	# Method #
	if ML_MAP:
		z = test(basis, weight_MAP)
	else:
		z = test(basis, weight_ML)



	# print MSE(z,t)

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