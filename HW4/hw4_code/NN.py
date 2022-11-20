import numpy as np 

'''
We are going to use the diabetes dataset provided by sklearn
https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
to train a 2 fully connected layer neural net. We are going to build the neural network from scratch.
'''


class dlnet:

    def __init__(self, x, y, lr = 0.01, batch_size=64, momentum=0.5, use_dropout=True, dropout_prob=0.3):
        '''
        This method initializes the class, it is implemented for you. 
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            alpha: slope coefficient for leaky relu
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

            momentum: coefficient for momentum-based update step
            change: dict of previous changes for each layer
        '''
        self.X=x # features
        self.Y=y # ground truth labels

        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [10, 15, 1] # dimensions of different layers
        self.alpha = 0.05
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        self.param = {} # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in back prop
        self.loss = [] # list to store loss values
        self.batch_y = [] # list of y batched numpy arrays

        self.iter = 0 # iterator to index into data for making a batch 
        self.batch_size = batch_size # batch size 
        
        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'classifier'
        self.neural_net_type = "Leaky Relu -> Tanh"

        self.momentum = momentum # momentum factor
        self.change = {} # dictionary for previous changes for momentum




    def nInit(self, param=None): 
        '''
        This method initializes the neural network variables, it is already implemented for you. 
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.
        '''
        if param is None:
            np.random.seed(1)
            self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
            self.param['b1'] = np.zeros((self.dims[1], 1))        
            self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
            self.param['b2'] = np.zeros((self.dims[2], 1))
        else:
            self.param = param

        for layer in self.param:
            self.change[layer] = np.zeros_like(self.param[layer])


    def Leaky_Relu(self,alpha, u):
        '''
        In this method you are going to implement element wise Leaky_Relu. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: 
            u of any dimension
            alpha: the slope coefficent of the negative part.
        return: Leaky_Relu(u) 
        '''
        # TODO: IMPLEMENT THIS METHOD
        return np.where(u > 0, u, alpha * u)

        # raise NotImplementedError
        

    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Do NOT use np.tanh. 
        Input: u of any dimension
        return: Tanh(u) 
        '''
        # TODO: IMPLEMENT THIS METHOD
        return (np.exp(u)-np.exp(-u))/(np.exp(u)+np.exp(-u))

        # raise NotImplementedError
    
    
    def dL_Relu(self,alpha, u):
        '''
        This method implements element wise differentiation of Leaky Relu, it is already implemented for you.  
        Input: 
             u of any dimension
             alpha: the slope coefficent of the negative part.
        return: dL_Relu(u) 
        '''
        return np.where(u > 0, 1.0, alpha)


    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh, it is already implemented for you.
        Input: u of any dimension
        return: dTanh(u) 
        '''
        return 1 - np.square(np.tanh(u))
    

    def nloss(self,y, yh):
        '''
        In this method you are going to implement mean squared loss. 
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        return: MSE 1x1: loss value 
        '''
        # TODO: IMPLEMENT THIS METHOD
        return 1/(2*y.shape[1]) * np.sum((y-yh)**2)
        
        # raise NotImplementedError


    @staticmethod
    def _dropout(u, prob):
        '''
        This method implements the dropout layer. Refer to the description for implementation details.
        Input: u D x N: input to dropout layer
        return: u_after_dropout D x N
                dropout_mask DxN
        '''
        # TODO: IMPLEMENT THIS METHOD
        mask = np.random.choice(2, size=u.shape, replace=True, p = [prob, 1-prob])
        return u * mask / (1-prob), mask
        # raise NotImplementedError


    def forward(self, x, use_dropout=True):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        Do not change the lines followed by #keep. 

        Input: x DxN: input to neural network
               use_dropout: True if using dropout in forward
        return: o2 1xN
        '''  
        # TODO: IMPLEMENT THIS METHOD

        self.ch['X'] = x #keep
        
        u1 = np.dot(self.param['theta1'], x) + self.param['b1'] # IMPLEMENT THIS LINE
        o1 = self.Leaky_Relu(self.alpha, u1) # IMPLEMENT THIS LINE

        if use_dropout:
            o1, dropout_mask = self._dropout(self.Leaky_Relu(self.alpha, u1), self.dropout_prob) # IMPLEMENT THIS LINE
            self.ch['u1'], self.ch['mask'], self.ch['o1'] = u1, dropout_mask, o1 #keep
        else:
            self.ch['u1'], self.ch['o1'] = u1, o1 #keep

        u2 = np.dot(self.param['theta2'], o1) + self.param['b2'] # IMPLEMENT THIS LINE
        o2 = self.Tanh(u2) # IMPLEMENT THIS LINE
        self.ch['u2'], self.ch['o2'] = u2, o2 #keep

        return o2 #keep

    def compute_gradients(self, y, yh, use_dropout=False):
        '''
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Input:
            y: 1 x N numpy array, ground truth values
            yh: 1 x N numpy array, predicting outputs

        Output:
            dLoss: dictionary that maps layer names (strings) to gradients (numpy arrays)
        '''
        # TODO: IMPLEMENT THIS METHOD
        print(self.ch["o2"].shape)
        dLoss_o2 = 1/len(y)*(ch["o2"]-y) # IMPLEMENT THIS LINE
        dLoss_u2 = np.dot(dLoss_o2, 1-(self.Tanh(ch["u2"]))**2) # IMPLEMENT THIS LINE
        dLoss_theta2 = np.dot(dLoss_u2, ch["o1"]) # IMPLEMENT THIS LINE
        dLoss_b2 = dLoss_u2 # IMPLEMENT THIS LINE
        dLoss_o1 = np.dot(dLoss_u2, self.param["theta2"])  # IMPLEMENT THIS LINE
        
        if use_dropout:
            do1du1 = np.dot(mask * self.alpha, self.Leaky_Relu(self.alpha, ch["o1"]))
            dLoss_u1 = np.dot(dLoss_o1, ) # IMPLEMENT THIS LINE
        else:
            dLoss_u1 = np.dot(dLoss_o1, self.Leaky_Relu(self.alpha, ch["o1"])) # IMPLEMENT THIS LINE


        dLoss_theta1 = np.dot(dLoss_u1, ch["X"]) # IMPLEMENT THIS LINE
        dLoss_b1 = dLoss_u1 # IMPLEMENT THIS LINE
        
        dLoss = {'theta1': dLoss_theta1, 'b1': dLoss_b1, 'theta2': dLoss_theta2, 'b2': dLoss_b2}
        return dLoss


    def update_weights(self, dLoss, use_momentum=False):
        '''
        Update weights of neural network based on learning rate given gradients for each layer. 
        Can also use momentum to smoothen descent.
        
        Input:
            dLoss: dictionary that maps layer names (strings) to gradients (numpy arrays)

        Return:
            None

        HINT: both self.change and self.param need to be updated for use_momentum=True and 
        only self.param needs to be updated when use_momentum=False
        '''
        # TODO: IMPLEMENT THIS METHOD

        for layer in dLoss:
            if use_momentum:
                z = self.change # IMPLEMENT THIS LINE
                #print(z) 
                if layer == "theta2": # IMPLEMENT THIS LINE
                    z1 = self.momentum *z["theta2"] + dLoss["theta2"]
                    self.param["theta2"] -= self.lr * z1
                elif layer == "b2":
                    z1 = self.momentum *z["b2"] + dLoss["b2"]
                    self.param["b2"] -= self.lr * z1
                elif layer == "theta1":
                    z1 = self.momentum *z["theta1"] + dLoss["theta1"]
                    self.param["theta1"] -= self.lr * z1
                elif layer == "b1":
                    z1 = self.momentum *z["b1"] + dLoss["b1"]
                    self.param["b1"] -= self.lr * z1
            else:   
                if layer == "theta2": # IMPLEMENT THIS LINE
                    self.param["theta2"] -= self.lr * dLoss["theta2"]
                elif layer == "b2":
                    self.param["b2"] -= self.lr * dLoss["b2"]
                elif layer == "theta1":
                    self.param["theta1"] -= self.lr * dLoss["theta1"]
                elif layer == "b1":
                    self.param["b1"] -= self.lr * dLoss["b1"]
        # print(self.param)

    def backward(self, y, yh, use_dropout=True, use_momentum=False):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemented methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.  

        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        Return: dLoss_theta2 (1x15), dLoss_b2 (1x1), dLoss_theta1 (15xD), dLoss_b1 (15x1)

        Hint: make calls to compute_gradients and update_weights
        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError


    def gradient_descent(self, x, y, iter = 60000, use_momentum=False, local_test=False):
        '''
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them. 
        2. One iteration here is one round of forward and backward propagation on the complete dataset. 
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss

        Input: x DxN: input
               y 1xN: labels
               iter: scalar, number of epochs to iterate through
        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError
       
    
    #bonus for undergraduate students 
    def batch_gradient_descent(self, x, y, iter = 60000, use_momentum=False, local_test=False):
        '''
        This function is an implementation of the batch gradient descent algorithm

        Notes: 
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient 
        2. One iteration here is one round of forward and backward propagation on one minibatch. 
           You will use self.iter and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations. 
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th, 
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the 
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail).

        Input: x DxN: input
               y 1xN: labels
               iter: scalar, number of BATCHES to iterate through
               local_test: boolean, True if calling local test, default False for autograder and Q1.3 
                    this variable can be used to switch between autograder and local test requirement for
                    appending/printing out loss and y batch arrays

        '''
        # TODO: IMPLEMENT THIS METHOD
        
        raise NotImplementedError


    def predict(self, x): 
        '''
        This function predicts new data points
        It is implemented for you

        Input: x DxN: inputs
        Return: y 1xN: predictions

        '''
        Yh = self.forward(x, use_dropout=False)
        return Yh
