import src.layers as layers
import numpy as np

convLayer = layers.ConvLayer(filters=3, kernel_size=(3,3), stride=1, padding=0)

X = np.array([[[1, 1, 0, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1, 0, 1, 0],
               [1, 1, 1, 1, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1]]])

kernel = np.array([[2, -1, 2],
                   [2, -1, 0],
                   [1, 0, 2]])

convLayer.setKernel(kernel)
result = convLayer.forward(X)
print("Convolution layer's result:\n", result)
expectedResult = np.array([[[4, 7, 1, 7, 2, 1,],
                            [6, 3, 5, 4, 4, 1,],
                            [6, 5, 6, 4, 4, 5,],
                            [4, 2, 5, 0, 6, -2,],
                            [5, 6, 6, 2, 5, 3,],
                            [2, 1, 2, 3, 2, 3,]]])
assert np.array_equal(result, expectedResult)

poolingLayer = layers.PoolingLayer(3, 3)

result = poolingLayer.forward(result)
expectedResult = np.array([[[7, 7], [6, 6]]])
print("Pooling layer's result:", result)
assert np.array_equal(result, expectedResult)

flattenLayer = layers.FlattenLayer()
result = flattenLayer.forward(result)
expectedResult = np.array([[7, 7, 6, 6]])
print("Flatten layer's result:", result)
assert np.array_equal(result, expectedResult)