from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
import time 
import multiprocessing 
import multiprocess, os
# from multiprocessing import Pool
from bo.bayesianOptimization.viz import vis_ei
from .internalBO import InternalBO
from ..gprInterface import GPR
from bo.gprInterface import InternalGPR
from ..sampling import uniform_sampling
from ..utils import compute_robustness
from ..behavior import Behavior

class RolloutEI(InternalBO):
    def __init__(self) -> None:
        pass

    def sample(
        self,
        test_function: Callable,
        horizon: int,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng
    ) -> Tuple[NDArray]:

        """Rollout with EI

        Args:
            test_function: Function of System Under Test.
            horizon: Number of steps to look ahead
            y_train: Evaluated values of samples from Training set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If y_train is not (n,) numpy array

        Returns:
            next x values that have minimum h step reward 
        """
        self.numthreads = int(multiprocessing.cpu_count()/2)
        self.tf = test_function
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        tf_dim = region_support.shape[0]
        num_samples = tf_dim*5 #* self.horizon  #* 10
        self.y_train = copy.deepcopy(np.asarray(test_function.point_history,dtype=object)[:,-1])

        # Choosing the next point
        x_opt = self._opt_acquisition(y_train, gpr_model, region_support, rng) 
        # Generate a sample dataset to rollout and find h step observations
        subx = uniform_sampling(5, region_support, tf_dim, rng)
        subx = np.vstack([subx,x_opt])
        # Rollout and get h step observations
        # print('-----',subx,region_support)
        suby = -1 * self.get_exp_values(subx)
        print('############################################################################')
        print()
        print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
        print()
        print('############################################################################')

        # Build Gaussian prior model with the sample dataset 
        for sample in tqdm(range(num_samples)):
            model = GPR(InternalGPR())
            model.fit(subx, suby)
            # Get the next point using EI 
            pred_sub_sample_x = self._opt_acquisition(suby, model, region_support, rng)  # EI for the outer BO 
            print('pred_sub_sample_x:  for {sample}: ',pred_sub_sample_x)
            # Rollout and get h step obs for the above point 
            vals = self.get_exp_values( pred_sub_sample_x)
            # print(vals)
            # vis_ei(pred_sub_sample_x[0],vals)
            pred_sub_sample_y = -1 * vals    # this uses the inner MC to rollout EI and get the accumulated the reward or yt+1
            print('pred_sub_sample_y:  for {sample} rolled out for 2 horizons : ',pred_sub_sample_y)
            print()
            # Stack to update the Gaussian prior
            subx = np.vstack((subx, pred_sub_sample_x))
            suby = np.hstack((suby, pred_sub_sample_y))
        # Find and return the next point with min obs val among this sample dataset 
        min_idx = np.argmin(suby)
        return subx[[min_idx],:]

    # Get expected value for each point after rolled out for h steps 
    def get_exp_values(self,eval_pts):
        # print('eval_pts: ',eval_pts,self.region_support.shape)
        eval_pts = eval_pts.reshape((-1,2)) # not needed for 1D
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        for i in range(num_pts):
            exp_val[i] = self.get_pt_reward(eval_pts[[i]],iters=5)
        return exp_val
            
    # def _evaluate_at_point_list(self, point_to_evaluate):
    #     self.point_current = point_to_evaluate
        
    #     if self.numthreads > 1:
    #         serial_mc_iters = [int(os.cpu_count()/self.numthreads)] * self.numthreads
    #         pool = multiprocess.Pool(processes=self.numthreads)
    #         rewards = pool.map_async(self.get_pt_reward, serial_mc_iters)
    #         print(rewards)
    #         rewards = rewards.get()
            
    #         # pool.close()
    #         # pool.join()
    #     else:
    #         rewards = self.get_pt_reward(self.point_current,self.mc_iters)
    #     return np.sum(rewards)/self.numthreads

    # Perform Monte carlo itegration
    def get_pt_reward(self,current_point,iters=int(5)):
        reward = 0
        for i in range(iters):
            reward += self.get_h_step_reward(current_point)
        return (reward/iters)
    
    # Rollout for h steps 
    def get_h_step_reward(self,current_point):
        reward = 0
        # Temporary Gaussian prior 
        tmp_gpr = copy.deepcopy(self.gpr_model)
        xtr = copy.deepcopy(np.asarray(self.tf.point_history,dtype=object)[:,1])  
        xtr = [i.tolist() for i in xtr]
        ytr = copy.deepcopy(self.y_train)
        h = self.horizon
        xt = current_point

        while(True):
            np.random.seed(int(time.time()))
            # print(xt)
            mu, std = self._surrogate(tmp_gpr, xt.reshape(1, -1))
            f_xt = np.random.normal(mu,std,1)
            ri = self.reward(f_xt,ytr)
            reward += ri
            h -= 1
            if h <= 0 :
                break
            
            xt = self._opt_acquisition(self.y_train,tmp_gpr,self.region_support,self.rng)
            np.append(xtr,[xt])
            np.append(ytr,[f_xt])
        return reward
    
    # Reward is the difference between the observed min and the obs from the posterior
    def reward(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        return r
