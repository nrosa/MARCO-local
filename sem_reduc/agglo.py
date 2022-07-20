import torch
import torch.nn.functional as F
import numpy as np

import tqdm

import cProfile
import time

class Agglomerative(object):

    def __init__(self,n_clusters=8, *, n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters	= n_clusters
        self.clusters = None
        self.batch_size = 32

        self.num_iterations = 0

    def _remove_empty_clusters(self):
        keep_cluster_idx = (self.cluster_dissimilarity > 9).sum(dim=-1) != self.cluster_dissimilarity.shape[0]

        # Del columns
        self.cluster_dissimilarity = self.cluster_dissimilarity[:,keep_cluster_idx]
        # Del rows
        self.cluster_dissimilarity = self.cluster_dissimilarity[keep_cluster_idx]

        # Update the cluster membership tensor
        # Set membership for all remain clusters
        temp_membership = self.cluster_membership.detach().clone()
        for i, idx in enumerate(torch.nonzero(keep_cluster_idx, as_tuple=False)):
            member_idx = torch.nonzero(temp_membership == idx, as_tuple=True)
            self.cluster_membership[member_idx] = i

        # Set membership for removed clusters
        for i, idx in enumerate(torch.nonzero(~keep_cluster_idx, as_tuple=False)):
            member_idx = torch.nonzero(temp_membership == idx, as_tuple=True)
            self.cluster_membership[member_idx] = self.cluster_dissimilarity.shape[0] + i





    def _init_cluster_dissimilarity(self, x):
        self.cluster_dissimilarity = []

        batch_size = 2048
        x1 = x
        for batch in x.split(batch_size):
            x2 = batch.transpose(-1, -2)
            # TODO test this calculation
            mags =  torch.matmul(
                x1.norm(p=2, dim=1).unsqueeze(1),
                x2.norm(p=2, dim=0).unsqueeze(0)
            )
            dissimilarity = 1 - (torch.matmul(x1, x2) / mags)
            del mags

            # Move to cpu as a 1080ti doesn't have enough vram
            self.cluster_dissimilarity.append(dissimilarity.to('cpu'))
        self.cluster_dissimilarity = torch.cat(self.cluster_dissimilarity, dim=1)

        # Set the diagonal to greater than the max disimilarity
        for i in range(self.cluster_dissimilarity.shape[0]):
            self.cluster_dissimilarity[i,i] = 10



        # Create the cluster membership tensor
        self.cluster_membership = torch.arange(x.shape[0])


    def _merge_clusters(self, c1_idx, c2_idx):
        # c1 will become the new cluster and c2 will be removed
        new_cluster, _ = torch.max(self.cluster_dissimilarity[(c1_idx,c2_idx),:], 0) 

        # c1 becomes the new cluster
        # Replace row
        self.cluster_dissimilarity[c1_idx] = new_cluster
        # Replace column
        self.cluster_dissimilarity[:,c1_idx] = new_cluster

        # Delete c2 from the array
        # Deleting takes too much time, just make larger than max dissimilarity
        # row_cond = torch.arange(self.cluster_dissimilarity.shape[0]) != c2_idx
        # self.cluster_dissimilarity = self.cluster_dissimilarity[row_cond, :]
        # self.cluster_dissimilarity = self.cluster_dissimilarity[:, row_cond]

        # Replace row  
        new_c2 = torch.ones_like(new_cluster) * 10
        self.cluster_dissimilarity[c2_idx] = new_c2
        # Replace column
        self.cluster_dissimilarity[:,c2_idx] = new_c2

        # Update cluster ids
        # Move c2 members into c1
        c2_member_idx = torch.nonzero(self.cluster_membership == c2_idx, as_tuple=True)
        self.cluster_membership[c2_member_idx] = c1_idx

        # Squeeze the id , only need to squeeze if clusters are deleted
        # tic = time.perf_counter()  
        # squeeze_idx = torch.nonzero(self.cluster_membership > c2_idx, as_tuple=True)
        # self.cluster_membership[squeeze_idx] -= 1
        # toc = time.perf_counter()
        # print(f"squeeze {toc - tic:0.4f} seconds")




    # def fit_predict(self, x):
    #     # Must have two dimension
    #     assert len(x.shape) == 2
    #     x= x.cuda()

    #     self.num_iterations = x.shape[0] - self.n_clusters

    #     self._init_cluster_dissimilarity(x)

    #     for j in tqdm.tqdm(range(self.num_iterations)):
    #     #for j in range(self.num_iterations):

    #         min_reduced = np.inf
    #         c1_idx = -1
    #         c2_idx = -1

    #         batch_size = 512
    #         for i, batch in enumerate(self.cluster_dissimilarity.split(batch_size)):
    #             batch = batch.cuda()

    #             reduced, cluster_diss_idx0 = batch.min(0)
    #             reduced, cluster_diss_idx1 = reduced.min(0)

    #             if reduced < min_reduced:
    #                 min_reduced = reduced
    #                 c1_idx = i * batch_size + cluster_diss_idx0[cluster_diss_idx1]
    #                 c2_idx = cluster_diss_idx1

    #         assert min_reduced != 10

    #         self._merge_clusters(c1_idx.to('cpu'), c2_idx.to('cpu'))

    #         if (j + 1) % 250 == 0:
    #             self._remove_empty_clusters()

    #     return self.cluster_membership


    def _find_min(self):
        min_reduced = np.inf
        c1_idx = -1
        c2_idx = -1

        batch_size = 512
        for i, batch in enumerate(self.cluster_dissimilarity.split(batch_size)):
            batch = batch.cuda()

            reduced, cluster_diss_idx0 = batch.min(0)
            reduced, cluster_diss_idx1 = reduced.min(0)

            if reduced < min_reduced:
                min_reduced = reduced
                c1_idx = i * batch_size + cluster_diss_idx0[cluster_diss_idx1]
                c2_idx = cluster_diss_idx1

        assert min_reduced != 10

        # Save the cluster ids so the clusters can be merged at a later time
        self.c1_idx = c1_idx.to('cpu')
        self.c2_idx = c2_idx.to('cpu')

        return min_reduced

    def init(self, x):
        assert len(x.shape) == 2
        x= x.cuda()
        self.num_iterations = x.shape[0] - self.n_clusters

        self._init_cluster_dissimilarity(x)
        
        return self._find_min()

    def step(self):
        self._merge_clusters(self.c1_idx, self.c2_idx)
        return self._find_min()


