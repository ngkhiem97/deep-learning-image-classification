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

convLayer1 = layers.Conv2DLayer(filters=6, kernel_size=(5, 5), stride=1, padding=0)
tanhLayer1 = layers.TanhLayer()
poolingLayer1 = layers.PoolingLayer(2, 2)
convLayer2 = layers.Conv3DLayer(filters=16, kernel_size=(5, 5), stride=1, padding=0)
reluLayer2 = layers.ReluLayer()
poolingLayer2 = layers.PoolingLayer(2, 2)
flattenLayer = layers.FlattenLayer()
fcLayer3 = layers.FullyConnectedLayer(2400, 120, xavier_init = True)
dropoutLayer3 = layers.DropoutLayer(0.8)
tanhLayer3 = layers.TanhLayer()
fcLayer4 = layers.FullyConnectedLayer(120, 84, xavier_init = True)
dropoutLayer4 = layers.DropoutLayer(0.8)
tanhLayer4 = layers.TanhLayer()
fcLayer5 = layers.FullyConnectedLayer(84, 10, xavier_init = True)
softmaxLayer = layers.SoftmaxLayer()
crossEntropyLoss = layers.CrossEntropy()
lenet = [convLayer1, tanhLayer1, poolingLayer1,
         convLayer2, reluLayer2, poolingLayer2, flattenLayer, 
         fcLayer3, dropoutLayer3, tanhLayer3, 
         fcLayer4, dropoutLayer4, tanhLayer4, 
         fcLayer5, softmaxLayer, crossEntropyLoss]

util.train_model(lenet, X_train, Y_train_encoded, X_test, Y_test_encoded, "lenet_xavier", 
                 learning_rate = 0.01, 
                 max_epochs = 20, 
                 batch_size = 10,
                 condition = 10e-10,
                 skip_first_layer=False)