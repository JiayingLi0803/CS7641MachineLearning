import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    #return np.asarray((X[:,0], X[:,1], np.exp( -0.2*np.abs(X[:,0]**2 + X[:,1]**2)))).T
    #return np.asarray((np.abs(X[:,0]), np.abs(X[:,1]), np.sqrt(2)*X[:,0] * X[:,1])).T
    return np.asarray((np.abs(X[:,0]), np.abs(X[:,1]), np.sqrt(2)*X[:,0] * X[:,1])).T
    # raise NotImplementedError


    
