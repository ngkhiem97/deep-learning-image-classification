import sys
from time import sleep
from tqdm import tqdm
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src2.Layers import FullyConnectedLayer, InputLayer, LinearLayer, ReluLayer, LogisticSigmoidLayer, TanhLayer, SoftmaxLayer, DropoutLayer, SquaredError, LogLoss, CrossEntropy

def batch_to_image(data):
    X_train = data
    #print("Shape before reshape:", X_train.shape)
    # Reshape the whole image data
    X_train = X_train.reshape(len(X_train), 3, 32, 32)
    #print("Shape after reshape and before transpose:", X_train.shape)
    # Transpose the whole data
    X_train = X_train.transpose(0,2,3,1)
    #print("Shape after reshape and transpose:", X_train.shape)
    return X_train

def display_image(image_index, batch,  label_names):
    # take first image
    image = batch[b'data'][image_index]
    # take first image label index
    label = batch[b'labels'][image_index]
    # Reshape the image
    image = image.reshape(3,32,32)
    # Transpose the image
    image = image.transpose(1,2,0)
    # Display the image
    plt.imshow(image)
    plt.title(label_names[label].decode())

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    data_batch_1 = unpickle("data_batch_1")
    meta_data = unpickle("batches.meta")
    label_names = meta_data[b'label_names']

    x = data_batch_1[b'data'] 
    y = np.array(data_batch_1[b'labels'])

    return x, y, label_names

def oneHotEncode(labels):
    distict_outputs = 10
    oneHot = np.zeros((labels.size, distict_outputs))
    oneHot[np.arange(labels.size), labels] = 1
    return oneHot

def fowardProp(x, layers, test=True, epoch=1):
    #forwards!
    h = x
    for i in range(len(layers)-1):
        if(isinstance(layers[i], DropoutLayer)):
            h = layers[i].forward(h, test, epoch)
        else:
            h = layers[i].forward(h)
    y_hat = h
    return y_hat

def backwardProp(y_train, y_hat, layers, epochs, lr,):
    grad = layers[-1].gradient(y_train, y_hat)
    for i in range(len(layers)-2,0,-1):
        new_grad = layers[i].backward(grad)
        if(isinstance(layers[i], FullyConnectedLayer)):
            layers[i].updateWeights(grad, epochs, lr)
        grad = new_grad

def test_split(data, frac= 2/3):
    two_thirds_rounded = math.ceil(frac * len(data))
    np.random.shuffle(data)
    train, test = data[:two_thirds_rounded,:], data[two_thirds_rounded:,:]
    return train, test

def trainModel(
    train, 
    test,
    x_slice,
    y_slice, 
    layers, 
    epochs_lim = 120, 
    lr = .0001, 
    loss_lim = math.pow(10,-5), 
    batch_num = 1
):
    #split into train and testing, last 10 rows are OHE labels

    loss = 10000
    train_metrics = []
    test_metrics = []

    epochs = 1 
    loss_delta = 1

    pbar = tqdm(total = epochs_lim, desc='Training Model', unit="Epoch")
    while(epochs <= epochs_lim):
        if (loss_delta > loss_lim):
            #break
            pass

        # epoch training loss
        x_train = train[:, x_slice]
        y_train = train[:, y_slice]
        y_hat = fowardProp(x_train, layers)

        new_loss = layers[-1].eval(y_train, y_hat)
        loss_delta = abs(loss - new_loss)
        #print(loss_delta)
        train_metrics.append(new_loss)
        loss = new_loss

        # epoch testing accuracy
        x_test = test[:, x_slice]
        y_test = test[:, y_slice]

        y_hat = fowardProp(x_test, layers)

        #acc = 0
        #for i in range(len(y_test)):
            #acc += y_test[i,np.argmax(y_hat[i])]

        test_loss = layers[-1].eval(y_test, y_hat)
        test_metrics.append(test_loss)


        #Shuffle
        prebatch = train
        np.random.shuffle(prebatch)
        batches = np.split(prebatch, batch_num)

        for mini_batch in batches:
            x_train = mini_batch[:, x_slice]
            y_train = mini_batch[:, y_slice]
            
            y_hat = fowardProp(x_train, layers, test=False, epoch=epochs)

            grad = layers[-1].gradient(y_train, y_hat)
            for i in range(len(layers)-2,0,-1):
                new_grad = layers[i].backward(grad)
                if(isinstance(layers[i], FullyConnectedLayer)):
                    layers[i].updateWeights(grad, epochs, lr)
                grad = new_grad

        epochs += 1
        pbar.update(1)
    pbar.close()
    x_train = train[:, x_slice]
    y_train = train[:, y_slice]
    # Final Accuracies
    model_Acc(x_train, y_train, layers, type = "Training")
    model_Acc(x_test, y_test, layers, type = "Testing")
    return layers, train_metrics, test_metrics


def model_Acc(x, y, layers, type = "Training"):
    y_hat = fowardProp(x, layers)
    acc = 0

    for i in range(len(y)):
        acc += y[i,np.argmax(y_hat[i])]

    acc = acc / len(y)
    print(f"{type} accuracy: {acc}")

def plotMetrics(train_loss, test_loss, title= "Graph"):
    plt.plot(train_loss, '-x', label="Train")
    plt.plot(test_loss, '-x', label="Test")
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('Cross entropy')
    plt.title(title)
    plt.show()


def MLP(epochs_lim, layer_1_size = 512, layer_2_size = 64):
    x_slice = slice(0, -10)
    y_slice = slice(-10, None)
    x, y, label_names = load_data()
    y_OHE = oneHotEncode(y)
    
    data = np.concatenate((x, y_OHE), axis =1)

    L1 = InputLayer(x)
    L2 = FullyConnectedLayer(x.shape[1], layer_1_size, xavier_init=True, Adam=True)
    L3 = ReluLayer()
    #L4 = DropoutLayer(0.8)
    L5 = FullyConnectedLayer(layer_1_size, layer_2_size, xavier_init=True, Adam=True)
    L6 = ReluLayer()
    L7 = DropoutLayer(0.4)
    L8 = FullyConnectedLayer(layer_2_size, y_OHE.shape[1], xavier_init=True, Adam=True)
    L9 = SoftmaxLayer()
    L10 = CrossEntropy()

    layers = [L1, L2, L3, L5, L6, L7 ,L8, L9, L10]
    train, test = test_split(data)
    
    layers, train_metrics, test_metrics = trainModel(
        train, 
        test, 
        x_slice, 
        y_slice, 
        layers, 
        epochs_lim, 
        lr=0.0001, 
        loss_lim=-1, 
        batch_num=59
    )
        
    plotMetrics(train_metrics, test_metrics)

    pass

def HW5():
    x_slice = slice(0, -10)
    y_slice = slice(-10, None)
    train_data = pd.read_csv('mnist_train_100.csv', header = None)
    test_data = pd.read_csv('mnist_valid_10.csv', header = None)
   
    x = test_data.values[:,1:]
    y = test_data.values[:,0]
    y_OHE = oneHotEncode(y)
    test_data = np.concatenate((x, y_OHE), axis =1)
   
    x = train_data.values[:,1:]
    y = train_data.values[:,0]
    y_OHE = oneHotEncode(y)
    train_data = np.concatenate((x, y_OHE), axis =1)
    


    L1 = InputLayer(x)
    L2 = FullyConnectedLayer(x.shape[1], 512, xavier_init=True)
    L3 = ReluLayer()
    L4 = DropoutLayer(.6) 
    L5 = FullyConnectedLayer(512, y_OHE.shape[1], xavier_init=True)
    L6 = SoftmaxLayer()
    L7 = CrossEntropy()

    layers = [L1, L2, L3, L4, L5, L6, L7]
    
    layers, train_metrics, test_metrics = trainModel(
        train_data, 
        test_data, 
        x_slice, 
        y_slice, 
        layers, 
        epochs_lim=60, 
        lr=0.001, 
        loss_lim=-1, 
        batch_num=10
    )
    plotMetrics(train_metrics, test_metrics)

    pass


if __name__ == '__main__':

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == '1':
            #MLP(50, layer_1_size = 512, layer_2_size = 64)
            # Training accuracy: 0.4689 Testing accuracy: 0.4152

            #MLP(50, layer_1_size = 512 * 2 , layer_2_size = 64)
            # Training accuracy: 0.4941 Testing accuracy: 0.4236

            #MLP(50, layer_1_size = 512 * 2 * 2, layer_2_size = 64)
            # Training accuracy: 0.5269 Testing accuracy: 0.4335

            MLP(50, layer_1_size = 1024, layer_2_size = 128*2)
            # Training accuracy: 0.4024 Testing accuracy: 0.3600

        elif cmd == '2':
            HW5()
        elif cmd == '3':
            pass
        elif cmd == '4':
            pass

    else:
        MLP()
        pass
