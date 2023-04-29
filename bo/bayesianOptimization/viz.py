import matplotlib.pyplot as plt
import numpy as np
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from .constants import NAME, H
 

def vis_ei(x,ei_vals):
    (fig, ax) = plt.subplots(1, 2, figsize=(5, 5))
    # ei = ExpectedImprovement(gp, opt_domain)
    # ei_vals = ei.evaluate_at_point_list(x)

    # # We estimate EI2 using 100 MC iterations, using grid search of size 50 to maximize the inner EI
    # ei2 = RolloutEI(gp, opt_domain, horizon=2, opt_mode='grid', mc_iters=20, grid_size=100)
    # ei2_vals = ei2.evaluate_at_point_list(x)

    _ = ax[0].plot(x, ei_vals, '--g')
    # _ = ax[1].plot(x, ei2_vals, '--g')
    _ = ax[0].set_title('EI Acquisition')
    # _ = ax[1].set_title('EI2 Acquisition')
    plt.show()


    
def plot_obj(X,func,opt,xcord,ycord,init,nopred):
    xaxis = np.arange(xcord[0], xcord[1], 0.01)
    yaxis = np.arange(ycord[0], ycord[1], 0.01)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results = func(np.array([x,y]))
    # simulate a sample made by an optimization algorithm
    # seed(1)
    sample_x = X[:init,0] #r_min + rand(10) * (r_max - r_min)
    sample_y = X[:init,1] #r_min + rand(10) * (r_max - r_min)
    # create a filled contour plot with 50 levels and jet color scheme
    pyplot.figure(figsize=(5,5))
#     axis = figure.subplot(projection='3d')
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
#     figure = pyplot.figure(figsize=(10,10))
#     axis = figure.add_subplot(projection='3d')
#     axis.plot_surface(x, y, results, cmap='jet')
    
    # define the known function optima
    optima_x = opt
    # draw the function optima as a white star
    pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
    # plot the sample as black circles
    pyplot.plot(sample_x, sample_y, 'o', color='black')
    rollx = X[init:,0] #xyrec[:,0] #
    rolly = X[init:,1] #xyrec[:,1] #
    pyplot.plot(rollx, rolly, 'x', color='white')
    pyplot.colorbar()
    # show the plot
    plt.savefig(str(H)+'_'+NAME+'_boplots1d_'+str(init)+'_'+str(nopred)+'.png')
    pyplot.show()


def plot_1d(X,func,opt,xcord,ycord,init,nopred):
    r_min, r_max = xcord, ycord
    xtrain = X[:,0]
    sc = lambda x: func(x) 
    ytrain = X[:,1]
    x = np.linspace(xcord, ycord, 40)[:, None]
    # sample input range uniformly at 0.01 increments
    inputs = np.arange(r_min, r_max, 0.01)
    # y_gp = gp.mean(x)
    # y_var = np.sqrt(gp.variance(x))
    # compute targets
    results = func([inputs]) #func(inputs)
    # print(results,inputs,xtrain,ytrain)
    # create a line plot of input vs result
    pyplot.figure(figsize=(8,5))
    pyplot.plot(inputs, results)
    # pyplot.plot(x, y_gp, color='r')
    print("xtrain: ",xtrain[init:])
    pyplot.plot(xtrain[:init], ytrain[:init], 'k.', markersize=15)
    pyplot.plot(xtrain[init:], ytrain[init:], '+', markersize=15)

    # define optimal input value
    x_optima = opt
    # draw a vertical line at the optimal input
    # pyplot.axvline(x=x_optima, ls='--', color='red')
    # pyplot.fill_between(x[:, 0], y_gp - y_var, y_gp + y_var, color='m', alpha=0.25)
    pyplot.legend(['True Function','Observations','Pred_obs'])
    # show the plot
    pyplot.savefig(str(H)+'_'+NAME+'_boplots1d_'+str(init)+'_'+str(nopred)+'.png')
    pyplot.show()
    