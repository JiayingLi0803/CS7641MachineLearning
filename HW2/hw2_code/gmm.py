import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        exp_x = np.exp(logit - np.max(logit, axis=-1, keepdims=True))
        sm = exp_x/np.sum(exp_x, axis=-1, keepdims=True)
        return sm

        #raise NotImplementedError

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        exp_x = np.exp(logit - np.max(logit, axis=-1, keepdims=True))
        sm = np.sum(exp_x, axis=-1, keepdims=True)
        lse = np.log(sm)+np.max(logit, axis=-1, keepdims=True)
        return lse

        # raise NotImplementedError

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """
        component1 = 1/(2*np.pi)**(len(mu_i)/2) # constant
        component2 = (np.linalg.det(sigma_i))**(-1/2) # constant
        component31 = np.dot((points-mu_i),np.linalg.inv(sigma_i)) # N*D matrix
        component32 = np.multiply(component31.T,(points-mu_i).T) # D*N matrix
        component33 = np.sum(component32, axis=0) # 1*N matrix
        component3 = np.exp(-1/2*component33)
        return component1*component2*component3

        # raise NotImplementedError

    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        np.random.seed(5) #Do not remove this line!
        pi = np.zeros(self.K) + 1/self.K
        mu = self.points[np.random.choice(self.points.shape[0], self.K, replace=True)]
        sigma = np.array([np.eye(len(self.points[0]))]*self.K)
        return pi, mu, sigma
        # raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        if full_matrix is True:
            ll = np.zeros((len(self.points), len(pi)))
            
            points = self.points
            for i in range(len(mu)):
                mu_i = mu[i]
                sigma_i = sigma[i]
                normalPDF = self.multinormalPDF(points, mu_i, sigma_i)    
                ll[:,i]= np.log(pi[i])+np.log(normalPDF) # (N,) array
            return ll

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        #raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        if full_matrix is True:
            jointMatrix = self._ll_joint(pi, mu, sigma)

            return self.softmax(jointMatrix)

        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        # raise NotImplementedError

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        if full_matrix is True:

            Nk = np.sum(gamma, axis=0)
            for i in range(len(gamma[0])):# range k
                mu_new = np.dot(gamma[:,i],self.points) / Nk[i]
                print(mu_new.shape)
                sigma_new = np.zeros((len(self.points[0]), len(self.points[0])))
                for j in range(len(self.points)):
                    sigma_new += gamma[i,j]*np.outer(self.points[j,:]-mu_new, self.points[j,:]-mu_new)
                sigma_new = sigma_new/Nk[i]

                pi_new = Nk[i]/np.sum(gamma)
            return pi_new, mu_new, signa_new


        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        raise NotImplementedError

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)