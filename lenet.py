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

convLayer1 = layers.ConvLayer(filters=3, kernel_size=(3,3), stride=1, padding=0)
reluLayer1 = layers.ReluLayer()
poolingLayer1 = layers.PoolingLayer(3, 3)
flattenLayer1 = layers.FlattenLayer()
fcLayer1 = layers.FullyConnectedLayer(100, 10, xavier_init = True)
dropoutLayer1 = layers.DropoutLayer(0.5)
softmaxLayer = layers.SoftmaxLayer()
crossEntropyLoss = layers.CrossEntropy()

lenet = [convLayer1, reluLayer1, poolingLayer1, flattenLayer1, fcLayer1, dropoutLayer1, softmaxLayer, crossEntropyLoss]

# util.train_model(lenet, X_train, Y_train_encoded, X_test, Y_test_encoded, "lenet", 
#                  learning_rate = 0.001, 
#                  max_epochs = 10, 
#                  batch_size = 25,
#                  condition = 10e-10,
#                  skip_first_layer=False)

h = util.forward(lenet, X_train[:2])
print(h.shape)

