import src.util as util
from sklearn.model_selection import train_test_split
import src.layers as layers
import numpy as np


cifar10_data = util.unpickle("cifar-10-batches-py/data_batch_1")
cifar10_label_names = util.unpickle("cifar-10-batches-py/batches.meta")[b'label_names']
cifar10_grey_images, cifar10_labels = util.convert_images_to_gray(cifar10_data)
cifar10_grey_images /= 255 # normalize to [0,1]
for i in range(2, 6, 1):
    cifar10_batch_data = util.unpickle(f"cifar-10-batches-py/data_batch_{i}")
    cifar10_batch_grey_images, cifar10_batch_labels = util.convert_images_to_gray(cifar10_batch_data)
    cifar10_batch_grey_images /= 255 # normalize to [0,1]
    cifar10_grey_images = np.concatenate((cifar10_grey_images, cifar10_batch_grey_images), axis=0)
    cifar10_labels = np.concatenate((cifar10_labels, cifar10_batch_labels), axis=0)


cifa10_test = util.unpickle("cifar-10-batches-py/test_batch")
cifa10_test_grey_images, cifa10_test_labels = util.convert_images_to_gray(cifa10_test)
cifa10_test_grey_images /= 255 # normalize to [0,1]

print("Size of training data:", cifar10_grey_images.shape)
print("Size of validation data:", cifa10_test_grey_images.shape)

X_train = cifar10_grey_images
Y_train = cifar10_labels
X_test = cifa10_test_grey_images
Y_test = cifa10_test_labels

Y_train_encoded = util.one_hot_array(Y_train, 10)
Y_test_encoded = util.one_hot_array(Y_test, 10)

print("Size of training labels:", Y_train_encoded.shape)
print("Size of validation labels:", Y_test_encoded.shape)

convLayer1 = layers.Conv2DLayer(filters=6, kernel_size=(3, 3), stride=1, padding=0)
tanhLayer1 = layers.TanhLayer()
poolingLayer1 = layers.PoolingLayer(2, 2)
# convLayer2 = layers.Conv3DLayer(filters=16, kernel_size=(3, 3), stride=1, padding=0)
# tanhLayer2 = layers.TanhLayer()
# poolingLayer2 = layers.PoolingLayer(2, 2)
flattenLayer = layers.FlattenLayer()
# fcLayer3 = layers.FullyConnectedLayer(3456, 120, xavier_init = True)
# tanhLayer3 = layers.TanhLayer()
fcLayer4 = layers.FullyConnectedLayer(1350, 84, xavier_init = True)
tanhLayer4 = layers.TanhLayer()
fcLayer5 = layers.FullyConnectedLayer(84, 10, xavier_init = True)
softmaxLayer = layers.SoftmaxLayer()
crossEntropyLoss = layers.CrossEntropy()
lenet = [convLayer1, tanhLayer1, poolingLayer1, flattenLayer, 
        #  convLayer2, tanhLayer2, poolingLayer2, 
        #  fcLayer3, tanhLayer3, 
         fcLayer4, tanhLayer4, 
         fcLayer5, softmaxLayer, crossEntropyLoss]

util.train_model(lenet, X_train[:100], Y_train_encoded[:100], X_test[:20], Y_test_encoded[:20], "lenet", 
                 learning_rate = 0.01, 
                 max_epochs = 10, 
                 batch_size = 20,
                 condition = 10e-10,
                 skip_first_layer=False)