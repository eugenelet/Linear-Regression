from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

def plot_contour(x,y,z,fig_num):
	fig = plt.figure(fig_num)
	ax = fig.gca(projection='3d')

	xi = np.linspace(min(x), max(x))
	yi = np.linspace(min(y), max(y))

	X, Y = np.meshgrid(xi, yi)

	Z = griddata(x, y, z, xi, yi)



	cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
	ax.clabel(cset, fontsize=9, inline=1)

	return;

def my_plot_surface(x,y,z,fig_num):
	fig = plt.figure(fig_num)
	ax = fig.gca(projection='3d')

	xi = np.linspace(min(x), max(x))
	yi = np.linspace(min(y), max(y))

	X, Y = np.meshgrid(xi, yi)

	Z = griddata(x, y, z, xi, yi)


	surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
	                       linewidth=1, antialiased=True)
	
	ax.set_zlim3d(np.min(Z), np.max(Z))
	fig.colorbar(surf)

	return;