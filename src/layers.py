from abc import ABC, abstractmethod
import numpy as np

EPSILON = 1e-7

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
 
    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([np.dot(gradIn_i, grad_i) for gradIn_i, grad_i in zip(gradIn, grad)])

    @abstractmethod
    def forward(self,dataIn):
        pass

    @abstractmethod  
    def gradient(self):
        pass

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored

    def gradient(self):
        pass

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        return np.identity(self.getPrevIn().shape[1])

    def backward(self, gradIn):
        return gradIn

class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(np.maximum(0,dataIn))
        return self.getPrevOut()

    def gradient(self):
        diag = np.where(self.getPrevOut() > 0, 1, 0)
        return np.eye(len(self.getPrevOut()[0])) * diag[:, np.newaxis]

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(1 / (1 + np.exp(-dataIn)))
        return self.getPrevOut()
    
    def gradient(self):
        diag = self.getPrevOut() * (1 - self.getPrevOut()) + EPSILON
        return np.eye(len(self.getPrevOut()[0])) * diag[:, np.newaxis]

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        max = np.max(dataIn, axis = 1)[:, np.newaxis]
        self.setPrevOut(np.exp(dataIn - max) / np.sum(np.exp(dataIn - max), axis=1, keepdims=True))
        return self.getPrevOut()

    def gradient(self):
        out = self.getPrevOut()
        tensor = np.empty((0, out.shape[1], out.shape[1]))
        for row in out:
            grad = -(row[:, np.newaxis])*row
            np.fill_diagonal(grad, row*(1-row))
            tensor = np.append(tensor, grad[np.newaxis], axis = 0)
        return tensor

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        a = np.exp(dataIn)-np.exp(-dataIn)
        b = np.exp(dataIn)+np.exp(-dataIn)
        self.setPrevOut(a/b)
        return self.getPrevOut()
    
    def gradient(self):
        diag = 1 - self.getPrevOut() ** 2 + EPSILON
        return np.eye(len(self.getPrevOut()[0])) * diag[:, np.newaxis]

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut, xavier_init = True):
        super().__init__()
        if xavier_init:
            self.xavier_init(sizeIn, sizeOut)
        else:
            self.weights = np.random.uniform(-0.001, 0.001, (sizeOut, sizeIn)).T
            self.biases = np.random.uniform(-0.001, 0.001, (1, sizeOut))

        # accumulators for Adam optimizer
        self.weights_s = 0
        self.weights_r = 0
        self.biases_s = 0
        self.biases_r = 0

        self.decay_1 = 0.9
        self.decay_2 = 0.999
        self.stability = 10e-8

    def xavier_init(self, sizeIn, sizeOut):
        bound = np.sqrt(6/(sizeIn+sizeOut))
        self.weights = np.random.uniform(-bound, bound, (sizeOut, sizeIn)).T
        self.biases = np.random.uniform(-bound, bound, (1, sizeOut))

    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = weights
    
    def getBiases(self):
        return self.biases
    
    def setBiases(self, biases):
        self.biases = biases

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(np.dot(dataIn, self.weights) + self.biases)
        return self.getPrevOut()

    def gradient(self):
        return np.array([self.weights.T for i in range(len(self.getPrevIn()))])
    
    def updateWeights(self, gradIn, epoch, learning_rate = 0.0001):
        dJdw = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
        self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
        weights_update = (self.weights_s/(1-self.decay_1**(epoch+1))) / (np.sqrt(self.weights_r/(1-self.decay_2**(epoch+1))) + self.stability)
        
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        self.biases_s = self.decay_1 * self.biases_s + (1 - self.decay_1) * dJdb
        self.biases_r = self.decay_2 * self.biases_r + (1 - self.decay_2) * dJdb * dJdb
        biases_update = (self.biases_s/(1-self.decay_1**(epoch+1))) / (np.sqrt(self.biases_r/(1-self.decay_2**(epoch+1))) + self.stability)
        
        self.setWeights(self.getWeights() - learning_rate * weights_update)
        self.setBiases(self.getBiases() - learning_rate * biases_update)

class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding=0):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = self.init_kernel()
        self.strides = strides
        self.padding = padding

    def init_kernel(self):
        bound = np.sqrt(6/(self.filters*self.kernel_size[0]*self.kernel_size[1]))
        return np.random.uniform(-bound, bound, (self.filters, self.kernel_size[0], self.kernel_size[1]))

    def getKernel(self):
        return self.kernel
    
    def setKernel(self, kernel):
        self.kernel = kernel

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.convolve(dataIn))
        return self.getPrevOut()

    def convolve(self, dataIn):
        return np.array([self.convolve2D(dataIn[i], self.kernel, self.padding, self.strides) for i in range(len(dataIn))])

    def convolve2D(self, image, kernel, padding=0, strides=1):
        kernalHeight = kernel.shape[0]
        kernalWidth = kernel.shape[1]
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]

        # Define Output Dimensions
        outputWidth = int(((imgHeight - kernalHeight + 2 * padding) / strides) + 1)
        outputHeight = int(((imgWidth - kernalWidth + 2 * padding) / strides) + 1)
        output = np.zeros((outputWidth, outputHeight))

        # Apply Padding
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - kernalWidth:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - kernalHeight:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + kernalHeight, y: y + kernalWidth]).sum()
                    except:
                        break

        return output

    def gradient(self):
        pass

# Objective functions
class SquaredError():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #TODO
        return np.mean((Y - Yhat) ** 2)

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #TODO
        return -2*(Y - Yhat)

class LogLoss():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #TODO
        return np.mean(-(Y * np.log(Yhat + EPSILON) + (1 - Y) * np.log(1 - Yhat + EPSILON)))

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #TODO
        return -((Y - Yhat) / (Yhat * (1 - Yhat) + EPSILON))

class CrossEntropy():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #TODO
        return -np.mean(Y * np.log(Yhat + EPSILON))

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #TODO
        return -(Y / (Yhat + EPSILON))
