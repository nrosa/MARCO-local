import torch
import torch.nn.functional as F
import numpy as np

import tqdm

import cProfile
import time

class Kmeans(object):

    def __init__(self,n_clusters=8, *, n_init=10, max_iter=300, tol=0.0001):
        # TODO run multiple inits and return best clusters
        
        self.n_clusters	= n_clusters
        self.max_iter = max_iter

        self.num_iterations = 0


    def _init_cluster_means(self, x):
        # TODO Implement kmeans ++ init
        rng = np.random.default_rng()
        self.cluster_means = x[rng.choice(x.shape[0], self.n_clusters)].detach().clone()

        self.cluster_membership = torch.zeros(x.shape[0], device=x.device)

    def _assign_cluster_membership(self, x):
        dist = torch.cdist(x, self.cluster_means)
        self.cluster_membership = dist.argmin(dim=-1)


    def _calc_cluster_means(self, x):
        for i in range(self.n_clusters):
            new_mean = x[self.cluster_membership==i].mean(dim=0)
            self.cluster_means[i] = new_mean


    def fit_predict(self, x):
        # Must have two dimension
        assert len(x.shape) == 2
        x= F.normalize(x.cuda(), p=2, dim=-1)

        self._init_cluster_means(x)

        for _ in range(self.max_iter):
        #for _ in range(10):
            # TODO testing
            old_membership = self.cluster_membership.detach().clone()
            self._assign_cluster_membership(x)
            self._calc_cluster_means(x)

            diff_cnt = torch.sum(self.cluster_membership != old_membership)

            if diff_cnt == 0:
                break

        print(diff_cnt)
     


        return self.cluster_membership
            



