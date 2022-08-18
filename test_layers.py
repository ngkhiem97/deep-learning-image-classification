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

grad = np.array([[[-2, 0],
                  [6, -2]]])
poolingResult = poolingLayer.backward(grad)
expectedResult = np.array([[[ 0, -2,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0, -2,  0],
                            [ 0,  6,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0]]])
print("Pooling layer's gradient:\n", poolingResult)
assert np.array_equal(poolingResult, expectedResult)

flattenLayer = layers.FlattenLayer()
result = flattenLayer.forward(result)
expectedResult = np.array([[7, 7, 6, 6]])
print("Flatten layer's result:", result)
assert np.array_equal(result, expectedResult)

grad = np.array([[-2, 0, 6, -2]])
flattenResult = flattenLayer.backward(grad)
expectedResult = np.array([[[-2, 0],
                            [6, -2]]])
print("Flatten layer's gradient:", flattenResult)
assert np.array_equal(flattenResult, expectedResult)

e = np.array([[[0, -2, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, -2, 0],
               [0, 6, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]])

kernel_T = np.array([[2, 2, 1],
                   [-1, -1, 0],
                   [2, 0, 2]])

result = convLayer.backward(e)
expectedResult = np.array([[[ 0, -4,  0, -4,  0,  0,  0,  0],
                            [ 0,  0,  2,  2,  0,  0,  0,  0],
                            [ 0, -2, -4, -4,  0,  0,  0,  0],
                            [ 0,  0,  0,  0, -4,  0, -4,  0],
                            [ 0, 12,  0, 12,  0,  2,  2,  0],
                            [ 0,  0, -6, -6, -2, -4, -4,  0],
                            [ 0,  6, 12, 12,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0]]])
print("Convolution layer's result:\n", result)
assert np.array_equal(result, expectedResult)

convLayer.updateKernel(e, 0)
print("Convolution layer's kernel:\n", convLayer.kernel)