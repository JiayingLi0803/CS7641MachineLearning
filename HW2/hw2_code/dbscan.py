import numpy as np
from kmeans import pairwise_dist

class DBSCAN(object):
    
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
        
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        cluster_idx = np.zeros(len(self.dataset)) - 1
        visited_indices = set()
        C = -1
        #neighbor_indices = [0, 2, 5, 8]
        for Pidx in range(len(self.dataset)):
            if Pidx not in visited_indices:
                visited_indices.add(Pidx)
                neighborPts = self.regionQuery(Pidx)
                if len(neighborPts) < self.minPts:
                    cluster_idx[Pidx] = -1
                else:
                    C += 1
                    self.expandCluster(Pidx, neighborPts, C, cluster_idx, visited_indices)

        return np.array(cluster_idx)

        # raise NotImplementedError

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        cluster_idx[index] = C # add P to cluster C
        
        PPrimeIdex = 0
        while PPrimeIdex < len(neighborIndices): # for each P' in neighbor points
            if neighborIndices[PPrimeIdex] not in visitedIndices: # if P' is not visited
                visitedIndices.add(neighborIndices[PPrimeIdex]) # mark P' as visited
                neighborPrimeIndices = self.regionQuery(neighborIndices[PPrimeIdex])
                if len(neighborPrimeIndices) >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, neighborPrimeIndices), axis = 0)
                    _, orderidx = np.unique(neighborIndices, return_index=True)
                    neighborIndices = neighborIndices[np.sort(orderidx)]
            if cluster_idx[neighborIndices[PPrimeIdex]] < 0:
                cluster_idx[neighborIndices[PPrimeIdex]] = C
            PPrimeIdex += 1
        # raise NotImplementedError
        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        return np.argwhere(pairwise_dist(np.array([self.dataset[pointIndex]]), self.dataset)<=self.eps)[:,1]

        # raise NotImplementedError