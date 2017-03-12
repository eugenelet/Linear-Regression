import numpy as np

def gaussian_basis(x, num,var):
	x = np.asarray(x)

	range_x = max(x) - min(x)
	gap_x = range_x / num


	basis = np.ones(len(x))
	for i in range (0,num):
		basis = np.vstack((basis,np.exp(-0.5*(x - i*gap_x)**2/var)))

	return basis

# x: data on x-axis
# y: data on y-axis
# num: number of basis
# variance: variance of each basis
def gaussian_basis_2d(x, y, num, variance):

	x = np.asarray(x)
	y = np.asarray(y)

	# Range of both axis
	range_x = max(x) - min(x)
	range_y = max(y) - min(y)

	# Gaps between means
	gap_x = range_x / num
	gap_y = range_y / num

	# Bias term
	basis = np.ones(len(x))

	# Stack basis
	for i in range (0,num):
		basis = np.vstack((basis,np.exp(-0.5*(x - i*gap_x)**2/variance)))
	for i in range (0,num):
		basis = np.vstack((basis,np.exp(-0.5*(y - i*gap_y)**2/variance)))

	return basis

def polynomial_basis(x, y, order):

	# Stack basis
	x_basis = np.asarray(x)
	for i in range (2,order):
		x_basis = np.vstack((x_basis,x**i))
	y_basis = np.asarray(y)
	for i in range (2,order):
		y_basis = np.vstack((y_basis,y**i))

	# Stack bias onto basis
	basis = np.append(x_basis,y_basis, axis=0)
	bias_basis = np.ones(basis.shape[1])
	basis = np.vstack((bias_basis,basis))

	return basis 