import math
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn
  
    def setPrevOut(self, out):
        self.__prevOut = out
  
    def getPrevIn(self):
        return self.__prevIn
  
    def getPrevOut(self):
        return self.__prevOut

    @abstractmethod
    def backward(self, gradIn):
        pass

    @abstractmethod
    def forward(self,dataIn):
        pass
  
    @abstractmethod  
    def gradient(self):
        pass

class InputLayer(Layer):
    def __init__(self, dataIn, zscore = True, div = True):
        super().__init__()
        self.zscore = zscore
        self.div = div
        if (self.zscore):

            # collect info for zscoring
            self.stdX = np.std(dataIn, 0, ddof = 1)
            
            # For numeric stability, set any feature that has a standard deviation of zero to 1
            for i in range(len(self.stdX)):
                #no deviation lets normalize this feature
                if (self.stdX[i] == 0):
                    dataIn[:, i] = np.ones(len(dataIn[:, i]))
                    self.stdX[i] = 1
            
            # Collect means to reflect std clean up
            self.meanX = np.mean(dataIn, 0) 
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if(self.zscore):
            dataOut = np.empty_like(dataIn)
            # ZSCORE
            for column in range(dataIn.shape[1]):
                dataOut[:,column] = np.divide((dataIn[:,column] - self.meanX[column]), self.stdX[column])
        elif (self.div):
            dataOut = np.divide(dataIn, 255.0)
        else:
            dataOut = dataIn 
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut, xavier_init = False, Adam = True):
        super().__init__()
        self.Adam = Adam
        np.random.seed(0)
        self.W = np.random.uniform(low=-.0001, high=.0001, size=(sizeIn, sizeOut))
        self.b = np.random.uniform(low=-.0001, high=.0001, size=(1,sizeOut))

        self.sw = 0
        self.rw = 0
        self.sb = 0
        self.rb = 0
        self.p_1 = 0.9
        self.p_2 = 0.999
        self.delt = math.pow(10,-8)
        
        if (xavier_init):
            xav = np.sqrt(
                np.divide(6, sizeIn + sizeOut)
            )
            self.W = np.random.uniform(low=-xav, high=xav, size=(sizeIn, sizeOut))
            self.b = np.random.uniform(low=-xav, high=xav, size=(1,sizeOut))

    def getWeights(self):
        return self.W

    def setWeights(self, weights):
        self.W = weights
        return

    def getBias(self):
        return self.b

    def setBias(self, bias):
        self.b = bias
        return

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        h = (dataIn @ self.W) + self.b
        self.setPrevOut(h)
        return h

    def gradient(self):
        return self.W.T

    def backward(self, gradIn):
        gradOut = gradIn @ self.gradient()
        return gradOut
        
    def updateWeights(self, gradIn, epoch, eta = 0.0001):
        if (self.Adam):
            
            dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
            dJdb = np.sum(gradIn, axis = 0) / gradIn.shape[0]
            
            self.sw = (self.p_1 * self.sw) + ((1 - self.p_1) * dJdW)
            self.rw = (self.p_2 * self.rw) + ((1 - self.p_2) * (dJdW * dJdW))

            num = np.divide(
                self.sw,
                1 - math.pow(self.p_1, epoch)
            )
            denom = np.sqrt(
                np.divide(
                    self.rw, 
                    1 - math.pow(self.p_2, epoch)
                )
            ) + self.delt
            adam_w = - eta * num / denom
            
            self.sb = (self.p_1 * self.sb) + ((1 - self.p_1) * dJdb)
            self.rb = (self.p_2 * self.rb) + ((1 - self.p_2) * (dJdb * dJdb))

            num = np.divide(
                self.sb,
                1 - math.pow(self.p_1, epoch)
            )
            denom = np.sqrt(
                np.divide(
                    self.rb, 
                    1 - math.pow(self.p_2, epoch)
                )
            ) + self.delt
            adam_b = - eta * num / denom

            self.W += adam_w
            self.b += adam_b

        else:
            ## dJdh = gradIn, dhdW = self.getPrevIn().T 
            dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
            dJdb = np.sum(gradIn, axis = 0) / gradIn.shape[0]
            
            self.W -= eta * dJdW
            self.b -= eta * dJdb

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        tensor = np.ones_like(self.getPrevOut())
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, dataIn, test=True, epoch=1):
        self.setPrevIn(dataIn)
        if (test):
            self.setPrevOut(dataIn)
            return dataIn
        else:
            np.random.seed(epoch)
            self.dropOutKey = np.random.rand(dataIn.shape[0], dataIn.shape[1]) < self.keep_prob
            dataOut = np.multiply(dataIn, self.dropOutKey)
            dataOut = dataOut / self.keep_prob
            self.setPrevOut(dataOut)
            return dataOut

    def gradient(self):
        tensor = np.ones_like(self.dropOutKey) / self.keep_prob
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        grad = np.where(self.getPrevOut() > 0, 1, 0)
        tensor = grad
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataOut = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        a = .000000000000001
        tensor = (self.getPrevOut() * (1 - self.getPrevOut())) + a
        return tensor

    def backward(self, gradIn):
        gradOut = np.multiply(gradIn, self.gradient())
        return gradOut
        
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataOut = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        a = .000000000000001
        tensor = (1 - self.getPrevOut()**2) + a
        tensor = np.array(tensor)
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        
        if dataIn.ndim > 1:
            # we’re processing a bunch of inputs the denominator should compute the sum for each row
            dataOut = []
            for row in dataIn:
                row = row - max(row)
                denominator = sum(np.exp(row))
                dataOut.append(np.exp(row) / denominator)
        else:
            # Avoid over/underflow
            dataIn = dataIn - max(dataIn)

            # calc the sum of e^ inputs to make a probability distribution.
            denominator = sum(np.exp(dataIn))
            dataOut = np.exp(dataIn) / denominator

        
        dataOut = np.array(dataOut)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        a = .000000000000001
        tensor = []
        for row in self.getPrevOut():
            grad = np.diag(row) - row[np.newaxis].T.dot(row[np.newaxis])
            tensor.append(grad)
        
        tensor = np.array(tensor)
        return tensor

    def backward(self, gradIn):

        this_grad = self.gradient()
        gradOut = np.empty_like(gradIn)

        for i in range(this_grad.shape[0]):
            gradOut[i] = gradIn[i] @ this_grad[i]
            
        return gradOut

class SquaredError():
    def eval(self, Y, Yhat):
        J = 1 / Y.shape[0] * (Y - Yhat).T.dot((Y - Yhat))
        return J[0][0]

    def gradient(self, Y, Yhat):
        return - 2 * (Y - Yhat)

class LogLoss():
    def eval(self , Y, Yhat):
        # Wrapper fro numertic stabability to add the a
        a = .000000000000001
        J = - 1 / Y.shape[0] * (Y.T.dot(np.log(Yhat + a)) + (1 - Y).T.dot(np.log(1 - Yhat + a)))
        return J[0][0]

    def gradient(self, Y, Yhat):
        a = .000000000000001
        return - np.divide((Y - Yhat), Yhat * (1 - Yhat) + a)

class CrossEntropy():
    def eval(self , Y, Yhat) :
        a = .000000000000001
        J = - 1 / Y.shape[0] * np.trace(
            Y.T.dot(np.log(Yhat + a))
        )
        return J

    def gradient(self, Y, Yhat):
        a = .000000000000001
        return -np.divide(Y, (Yhat + a))