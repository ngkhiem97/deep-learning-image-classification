import src.layers as layers
import numpy as np

convLayer = layers.ConvLayer(filters=3, kernel_size=(3,3), strides=1, padding=0)

X = np.array([[[1, 1, 0, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1],
               [1, 1, 1, 1, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1]]])

kernel = np.array([[2, -1, 2],
                   [2, -1, 0],
                   [1, 0, 2]])

convLayer.setKernel(kernel)
result = convLayer.forward(X)

expectedResult = np.array([[[4, 7, 1, 7, 2, 1,],
                            [6, 5, 5, 5, 4, 3,],
                            [3, 2, 2, 5, 1, 3,],
                            [5, 6, 6, 2, 5, 3,],
                            [2, 1, 2, 3, 2, 3,]]])

assert np.array_equal(result, expectedResult)