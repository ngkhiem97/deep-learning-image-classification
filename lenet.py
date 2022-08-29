import src.util as util
from sklearn.model_selection import train_test_split
import src.layers as layers

cifar10_data = util.unpickle("cifar-10-batches-py/data_batch_1")
cifar10_label_names = util.unpickle("cifar-10-batches-py/batches.meta")[b'label_names']
cifar10_grey_images, cifar10_labels = util.convert_images_to_gray(cifar10_data)
cifar10_grey_images /= 255 # normalize to [0,1]

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(cifar10_grey_images, cifar10_labels,
                                                    stratify=cifar10_labels, test_size=0.2, random_state=42)

# convert the labels to one-hot vectors
Y_train_encoded = util.one_hot_array(Y_train, 10)
Y_test_encoded = util.one_hot_array(Y_test, 10)

# util.show_image(X_train[0], Y_train[0], cifar10_label_names)

convLayer1 = layers.Conv2DLayer(filters=3, kernel_size=(3, 3), stride=1, padding=0)
tanhLayer1 = layers.TanhLayer()
poolingLayer1 = layers.PoolingLayer(2, 2)
convLayer2 = layers.Conv3DLayer(filters=16, kernel_size=(5, 5), stride=1, padding=0)
reluLayer2 = layers.ReluLayer()
poolingLayer2 = layers.PoolingLayer(2, 2)
flattenLayer = layers.FlattenLayer()
fcLayer3 = layers.FullyConnectedLayer(2400, 120, xavier_init = True)
dropoutLayer3 = layers.DropoutLayer(0.8)
tanhLayer3 = layers.TanhLayer()
fcLayer4 = layers.FullyConnectedLayer(675, 42, xavier_init = True)
dropoutLayer4 = layers.DropoutLayer(0.8)
tanhLayer4 = layers.TanhLayer()
fcLayer5 = layers.FullyConnectedLayer(42, 10, xavier_init = True)
softmaxLayer = layers.SoftmaxLayer()
crossEntropyLoss = layers.CrossEntropy()

lenet = [convLayer1, tanhLayer1, poolingLayer1, flattenLayer, 
        #  convLayer2, reluLayer2, poolingLayer2, 
        #  fcLayer3, dropoutLayer3, tanhLayer3, 
         fcLayer4, dropoutLayer4, tanhLayer4, 
         fcLayer5, softmaxLayer, crossEntropyLoss]

util.train_model(lenet, X_train[:1000], Y_train_encoded[:1000], X_test[:100], Y_test_encoded[:100], "lenet", 
                 learning_rate = 0.1, 
                 max_epochs = 100, 
                 batch_size = 100,
                 condition = 10e-10,
                 skip_first_layer=False)

# h = util.forward(lenet, X_train[:2])
# h = lenet[-1].forward(h)
# print(h.shape)

# h = convLayer1.forward(X_train[:1])
# h = tanhLayer1.forward(h)
# h = poolingLayer1.forward(h)
# h = convLayer2.forward(h)
# h = reluLayer2.forward(h)
# h = poolingLayer2.forward(h)
# h = flattenLayer.forward(h)
# h = fcLayer3.forward(h)
# h = dropoutLayer3.forward(h)
# h = tanhLayer3.forward(h)
# h = fcLayer4.forward(h)
# h = dropoutLayer4.forward(h)
# h = tanhLayer4.forward(h)
# h = fcLayer5.forward(h)
# h = softmaxLayer.forward(h)

# grad = crossEntropyLoss.gradient(Y_train_encoded[:1], h)
# grad = softmaxLayer.backward(grad)
# grad = fcLayer5.backward(grad)
# grad = tanhLayer4.backward(grad)
# grad = dropoutLayer4.backward(grad)
# grad = fcLayer4.backward(grad)
# grad = tanhLayer3.backward(grad)
# grad = dropoutLayer3.backward(grad)
# grad = fcLayer3.backward(grad)
# grad = flattenLayer.backward(grad)
# grad = poolingLayer2.backward(grad)
# grad = reluLayer2.backward(grad)
# convLayer2.updateKernel(grad, 1, 0.1)
# grad = convLayer2.backward(grad)
# grad = poolingLayer1.backward(grad)
# grad = tanhLayer1.backward(grad)
# convLayer1.updateKernel(grad, 1, 0.1)

# print(grad.shape)