'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''
import numpy as np

class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [2 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.

        # Method 1: Cannot pass localtests or Gradescope
        return points[np.random.choice(points.shape[0], K, replace=False)]

        # Method 3: Cannot pass localtests or Gradescope
        return points[np.random.uniform(0,len(points),K).astype("int")]
        """
        # Method 2: Pass localtests, but cannot pass Gradescope
        np.random.seed(1)
        new = np.random.permutation(points)
        return points[np.random.choice(len(points), K, replace=False)]
        
        # raise NotImplementedError

    def _kmpp_init(self, points, K, **kwargs): # [3 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        np.random.seed()
        sample_points = points[np.random.choice(points.shape[0], len(points)//100+1, replace = False)]
        initidx = np.random.randint(len(sample_points),size = 1)[0]
        cluster_centers = np.copy([sample_points[initidx]])
        
        sample_points = np.delete(sample_points, initidx, 0)
        if K == 1:
            return cluster_centers
        for kidx in range(1, K):
            maxDist = 0
            next_center = sample_points[0]
            for idx in range(len(sample_points)):
                dist = np.min(np.sum(np.square(sample_points[idx]-cluster_centers),axis = 1))
                #np.square(sample_points[idx] - cluster_centers[kidx-1])
                if dist > maxDist:
                    next_center = sample_points[idx]
            cluster_centers = np.vstack([cluster_centers, next_center])
            sample_points = np.delete(sample_points, idx, 0)
        return cluster_centers
        # raise NotImplementedError

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        return np.argmin(pairwise_dist(points, centers), axis = 1)
        #raise NotImplementedError

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting! 
        dictionary
        d = dict()
        for i in range(np.max(cluster_idx)+1):
            d[i] = []
        for i in range(len(cluster_idx)):
            d[cluster_idx[i]].append(points[i])
        """
        points = points.astype('float64')
        centers_mapping = {key : [] for key in [i for i in range(old_centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])

        l = []
        for i in range(np.max(cluster_idx)+1):
            l.append(np.mean(centers_mapping[i],axis = 0))
        
        return np.array(l)
        
        # raise NotImplementedError
        
    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """

        sum = 0
        for i in range(len(points)):
            sum += np.sum(np.square((centers[cluster_idx[i]])-(points[i])))
        return sum

        # raise NotImplementedError

    def _get_centers_mapping(self, points, cluster_idx, centers):
        # This function has been implemented for you, no change needed.
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping   



    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        This function has been implemented for you, no change needed.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            #print("iteration: ", it)
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            #print("points: ",points)
            #print("cluster_idx: ", cluster_idx)
            #print("centers: ", centers)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                # print("iteration: ", it)
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss

def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    
    x2 = np.sum(x**2, axis=1) # shape of (m)
    y2 = np.sum(y**2, axis=1) # shape of (n)

    xy = np.matmul(x, y.T)

    x2 = x2.reshape(-1, 1)
    dists = np.sqrt(x2 - 2*xy + y2)
    return dists
    """
    return np.sqrt(np.sum(np.square(x)[:,np.newaxis,:], axis=2) - 2 * x.dot(y.T) + np.sum(np.square(y), axis=1))

    # return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    # raise NotImplementedError

def silhouette_coefficient(points, cluster_idx, centers, centers_mapping): # [10pts]
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """
    
    SCarray, mu_ins, mu_outs = np.zeros(len(points)), np.zeros(len(points)), np.zeros(len(points))
    for point_idx in range(len(points)): # for each point
        point_cluster = cluster_idx[point_idx]
        mu_out_array = np.zeros(len(centers_mapping))
        for rest_cluster in centers_mapping:
            if rest_cluster != point_cluster: # different cluster
                rest_cluster_points = centers_mapping[rest_cluster]
                nj = len(rest_cluster_points)
                dist_rest_cluster = np.sum(pairwise_dist(rest_cluster_points,  np.array([points[point_idx]])))
                mu_out = dist_rest_cluster/nj
                mu_out_array[rest_cluster] = mu_out
            else:
                mu_out_array[rest_cluster] = float("Inf")
                self_cluster_points = centers_mapping[point_cluster]
                if len(self_cluster_points) == 1: # if only one point in self cluster
                    mu_in = 0
                else:
                    in_cluster_idx = np.where(self_cluster_points==points[point_idx][0])[0]
                    mu_in = np.sum(pairwise_dist(self_cluster_points, np.array([points[point_idx]])))/(len(self_cluster_points)-1)

        mu_out_min = min(mu_out_array)
        si = (mu_out_min-mu_in)/(max(mu_out_min, mu_in)) # calculate s_i

        SCarray[point_idx] = si
        mu_ins[point_idx] = mu_in
        mu_outs[point_idx] = mu_out_min

    return np.sum(SCarray)/len(SCarray), np.array([mu_ins]), np.array([mu_outs])
    # raise NotImplementedError