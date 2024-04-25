
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
import pdb


############## MLP functions ##################################################################################################################################################################
###############################################################################################################################################################################################
def F_standardize(X):
    """
    standardize X, i.e. subtract mean (over data) and divide by standard-deviation (over data)

    Parameters
    ----------
    X: np.array of size (nbData, nbDim)
        matrix containing the observation data

    Returns
    -------
    X: np.array of size (nbData, nbDim)
        standardize version of X
    """

    X -= np.mean(X, axis=0, keepdims=True)
    X /= (np.std(X, axis=0, keepdims=True) + 1e-16)
    return X

def F_sigmoid(x):
    """Compute the value of the sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def F_relu(x):
    """Compute the value of the Rectified Linear Unit activation function"""
    return x * (x > 0)

def F_dRelu(x):
    """Compute the derivative of the Rectified Linear Unit activation function"""
    ## --- START CODE HERE
    y = (x > 0)
    # --- END CODE HERE
    return y



def F_computeCost(hat_y,y):
    """Compute the cost (sum of the losses)

    Parameters
    ----------
    hat_y: (1, nbData)
        predicted value by the MLP
    y: (1, nbData)
        ground-truth class to predict
    """
    m = hat_y.shape[1]

    # --- START CODE HERE
    #print('computing loss...')
    #print(y)
    #print(hat_y)
    loss = -y * np.log(hat_y) - (1-y) * np.log(1-hat_y)
    #print(loss)
    # --- END CODE HERE

    cost = np.sum(loss) / m
    return cost

def F_computeAccuracy(hat_y,y):
    """Compute the accuracy

    Parameters
    ----------
    hat_y: (1, nbData)
        predicted value by the MLP
    y: (1, nbData)
        ground-truth class to predict
    """

    m = y.shape[1]
    class_y = np.copy(hat_y)
    class_y[class_y>=0.5]=1
    class_y[class_y<0.5]=0
    return np.sum(class_y==y) / m


############# Objects #################################################################################################################################################################
#########################################################################################################################################################################################

class C_MultiLayerPerceptron:
    """
    A class used to represent a Multi-Layer Perceptron with 1 hidden layers

    ...

    Attributes
    ----------
    W1, b1, W2, b2:
        weights and biases to be learnt
    Z1, A1, Z2, A2:
        values of the internal neurons to be used for backpropagation
    dW1, db1, dW2, db2, dZ1, dZ2:
        partial derivatives of the loss w.r.t. parameters

    Methods
    -------
    forward_propagation

    backward_propagation

    update_parameters

    """

    W1, b1, W2, b2 = [], [], [], []
    Z1, A1, Z2, A2 = [], [], [], []
    dW1, db1, dW2, db2 = [], [], [], []
    dZ1, dA1, dZ2 = [], [], []

    def __init__(self, n_in, n_h, n_out):
        #initialise weight and biases parameters
        #we take random weights and null biais to start with
        self.W1 = np.random.randn(n_h, n_in) * 0.01
        self.b1 = np.zeros(shape=(n_h, 1))
        self.W2 = np.random.randn(n_out, n_h) * 0.01
        self.b2 = np.zeros(shape=(n_out, 1))
        return

    def __setattr__(self, attrName, val):
        if hasattr(self, attrName):
            self.__dict__[attrName] = val
        else:
            raise Exception("self.%s note part of the fields" % attrName)



    def M_forwardPropagation(self, X):
        """Forward propagation in the MLP
        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data

        Return
        ------
        hat_y: numpy array (1, nbData)
            predicted value by the MLP
        """

        # --- START CODE HERE
        self.Z1 = np.dot(self.W1,X) + self.b1
        self.A1 = F_relu(self.Z1)
        self.Z2 = np.dot(self.W2,self.A1) + self.b2
        self.A2 = F_sigmoid(self.Z2)
        # --- END CODE HERE

        hat_y = self.A2

        return hat_y


    def M_backwardPropagation(self, X, y):
        """Backward propagation in the MLP
        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data
        y: numpy array (1, nbData)
            ground-truth class to predict
        """

        m = y.shape[1]  #batch size

        # --- START CODE HERE
        self.dZ2 = self.A2 - y
        self.dW2 = np.dot(self.dZ2,(self.A1).T)/m
        self.db2 = np.sum(self.dZ2, axis=1,keepdims = True)/m
        self.dA1 = np.dot(self.W2.T,self.dZ2)
        self.dZ1 = np.multiply(self.dA1,F_dRelu(self.Z1))  # pointwise product with multiply
        self.dW1 = np.dot(self.dZ1,X.T)/m
        self.db1 = np.sum(self.dZ1, axis=1, keepdims = True)/m
        # --- END CODE HERE
        return


    def M_gradientDescent(self, alpha):
        """Update the parameters of the network using gradient descent

        Parameters
        ----------
        alpha: float scalar
            amount of update at each step of the gradient descent

        """

        # --- START CODE HERE
        self.W1 -= alpha * self.dW1
        self.b1 -= alpha * self.db1
        self.W2 -= alpha * self.dW2
        self.b2 -= alpha * self.db2
        # --- END CODE HERE

        return





############# Data #################################################################################################################################################################
#########################################################################################################################################################################################



print("Récupération de la base de données...")
X, y = datasets.make_circles(n_samples=1000, noise=0.05, factor=0.5)

print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(set(y))

# X is (nbExamples, nbDim)
# y is (nbExamples,)

# --- Standardize data
X = F_standardize(X)

# --- Split between training set and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# --- Convert to proper shape: (nbExamples, nbDim) -> (nbDim, nbExamples)
X_train = X_train.T
X_test = X_test.T

# --- Convert to proper shape: (nbExamples,) -> (1, nbExamples)
y_train = y_train.reshape(1, len(y_train))
y_test = y_test.reshape(1, len(y_test))

n_in = X_train.shape[0]
n_out = 1

print("X_train.shape: {}".format(X_train.shape))
print("X_test.shape: {}".format(X_test.shape))
print("y_train.shape: {}".format(y_train.shape))
print("y_test.shape: {}".format(y_test.shape))
print("n_in: {} n_out: {}".format(n_in, n_out))



# plot data
plt.scatter(X_train[0,np.ravel(y_train==0)],X_train[1,np.ravel(y_train==0)],color='r')
plt.scatter(X_train[0,np.ravel(y_train==1)],X_train[1,np.ravel(y_train==1)],color='b')
plt.show()
# Basicaly, we are training a MLP to tell, given the coordinates
# of a point,  whether this point belongs to the tiny (blue, class 1) 
# or the big circle (red, class 0).

print("we just plotted the data, we can see that it is not linearly separable")
input("Press Enter to continue and see the results of the MLP training...")



############# MLP #################################################################################################################################################################
#########################################################################################################################################################################################

# Instantiate the class MLP with providing
# the size of the various layers (input=4, hidden=10, outout=1)

n_hidden = 10
num_epoch = 5000


myMLP = C_MultiLayerPerceptron(n_in, n_hidden, n_out)

train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []

# Run over epochs
for i in range(0, num_epoch):
    print('epoch: ',i)
    # --- Forward
    y_predict_train = myMLP.M_forwardPropagation(X_train)

    # --- Store results on train
    train_cost.append( F_computeCost(y_predict_train, y_train) )
    train_accuracy.append( F_computeAccuracy(y_predict_train, y_train) )

    # --- Backward
    myMLP.M_backwardPropagation(X_train, y_train)

    # --- Update
    myMLP.M_gradientDescent(alpha=0.1)

    # --- Store results on test
    y_predict_test = myMLP.M_forwardPropagation(X_test)
    test_cost.append( F_computeCost(y_predict_test, y_test) )
    test_accuracy.append( F_computeAccuracy(y_predict_test, y_test) )

    if (i % 100)==0:
        print("epoch: {0:d} (cost: train {1:.2f} test {2:.2f}) (accuracy: train {3:.2f} test {4:.2f})".format(i, train_cost[-1], test_cost[-1], train_accuracy[-1], test_accuracy[-1]))





plt.subplot(1,2,1)
plt.plot(train_cost, 'r')
plt.plot(test_cost, 'g--')
plt.xlabel('# epoch')
plt.ylabel('loss')
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(train_accuracy, 'r')
plt.plot(test_accuracy, 'g--')
plt.xlabel('# epoch')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()












