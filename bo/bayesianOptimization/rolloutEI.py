from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
import copy
import time 

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
        # num_samples: int,
        # x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: Callable,
        rng,
        behavior:Behavior = Behavior.MINIMIZATION
    ) -> Tuple[NDArray]:

        """Internal BO Model

        Args:
            test_function: Function of System Under Test.
            num_samples: Number of samples to generate from BO.
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_complete
            y_complete
            x_new
            y_new
        """
        self.tf = test_function
        self.gpr_model = gpr_model
        self.horizon = horizon
        self.region_support = region_support
        self.rng = rng
        falsified = False
        self.behavior = behavior
        tf_dim = region_support.shape[0]
        num_samples = tf_dim*5

        self.y_train = copy.deepcopy(np.asarray(test_function.point_history)[:,-1])
        x_opt = self._opt_acquisition(y_train, gpr_model, region_support, rng) 
        subx = uniform_sampling(5, region_support, tf_dim, rng)
        subx = np.vstack([subx,x_opt])
        # print('subx: ',subx)
        suby = -1 * self.get_exp_values(subx)
        print('############################################################################')
        print()
        print('suby: rollout for 2 horizons with 6 sampled points  :',suby, ' subx:', subx)
        print()
        print('############################################################################')

        for sample in tqdm(range(num_samples)):
            model = GPR(InternalGPR())
            # print(dir(model))
            model.fit(subx, suby)

            pred_sub_sample_x = self._opt_acquisition(suby, model, region_support, rng)  # EI for the outer BO 
            print('pred_sub_sample_x:  for {sample}: ',pred_sub_sample_x)
            pred_sub_sample_y = -1 * self.get_exp_values( pred_sub_sample_x)    # this uses the inner MC to rollout EI and get the accumulated the reward or yt+1
            print('pred_sub_sample_y:  for {sample} rolled out for 2 horizons : ',pred_sub_sample_y)
            print()
            subx = np.vstack((subx, pred_sub_sample_x))
            suby = np.hstack((suby, pred_sub_sample_y))

        # print('Inside rollout sample')
        min_idx = np.argmin(suby)
        return subx[[min_idx],:]


    def get_exp_values(self,eval_pts,iters=int(5)):
        # super().__init__(gpr_model)
        # self.gpr_model = gpr_model
        
        self.iters = iters
        # print('shap eval',eval_pts,eval_pts.shape)
        eval_pts = eval_pts.reshape((-1,2))
        num_pts = eval_pts.shape[0]
        exp_val = np.zeros(num_pts)
        # print('eval pts : ',eval_pts,eval_pts.shape)
        for i in range(num_pts):
            exp_val[i] = self.get_pt_reward(eval_pts[[i],:])
        # print('EXP val :',exp_val, i)
        return exp_val
            

    def get_pt_reward(self,current_point):
        reward = 0
        for i in range(self.iters):
            reward += self.get_h_step_reward(current_point)
        return (reward/self.iters)
    
    def get_h_step_reward(self,current_point):
        reward = 0
        tmp_gpr = copy.deepcopy(self.gpr_model)
        # print(tmp_gpr,self.gpr_model)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)           
        xtr = copy.deepcopy(np.asarray(self.tf.point_history)[:,1])  
        xtr = [i.tolist() for i in xtr]
        # print('xtr: -----------',xtr)
        ytr = copy.deepcopy(self.y_train)
        # print('ADDRESS: ',id(ytr), 'self.tf.point_history: ',(id(self.tf.point_history)))
        h = self.horizon
        xt = current_point

        while(True):
            np.random.seed(int(time.time()))
            # print(xt.reshape(1, -1))
            mu, std = self._surrogate(tmp_gpr, xt.reshape(1, -1))
            f_xt = np.random.normal(mu,std,1)
            ri = self.reward(tmp_gpr,f_xt,ytr)
            reward += ri
            print('fxt: ,',f_xt,'h: ',h)
            h -= 1
            if h <= 0 :
                break
            
            xt = self._opt_acquisition(self.y_train,tmp_gpr,self.region_support,self.rng)
            # ypred, _ = compute_robustness(np.array([xt]), self.tf, self.behavior)
            np.append(xtr,[xt])
            np.append(ytr,[f_xt])
        return reward
    

    def reward(self,gpr,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        return r
