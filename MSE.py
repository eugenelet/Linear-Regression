
def MSE(test_res, target):
	error = 0
	for i in range(0,len(test_res)):
		error += (test_res[i] - target[i])**2
	return error / (2*len(test_res))
