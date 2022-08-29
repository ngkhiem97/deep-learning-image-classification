import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import src.layers as layers

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_image(image, label, label_names):
    plt.imshow(image)
    plt.title(label_names[label].decode())
    plt.show()

def convert_images_to_gray(batch):
    converted_images = np.zeros((batch[b'data'].shape[0], 32, 32, 1))
    for i in range(len(batch[b'data'])):
        image = batch[b'data'][i]
        image = image.reshape(3,32,32)
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.reshape(32,32,1)
        converted_images[i] = image
    return converted_images, batch[b'labels']

def one_hot(y, n_classes):
    return np.eye(n_classes).astype(float)[y]

def one_hot_array(y, n_classes):
    return np.array([one_hot(y_i, n_classes) for y_i in y])

def decode(y_pred):
    return np.array([np.argmax(y_i) for y_i in y_pred]) 

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def forward(layers, X):
    h = X
    for layer in layers[:-1]:
        h = layer.forward(h)
    return h

def train_model(layers_, X_train, Y_train, X_val, Y_val, filename="default", learning_rate = 0.001, max_epochs = 100, batch_size = 25, condition = 10e-10, skip_first_layer = True):
    epoch = 0
    lastEval = 0
    loss_train = []
    loss_val = []
    batch_size = 25

    pbar = tqdm(total = max_epochs, desc='Training Model', unit="Epoch")
    while (epoch < max_epochs):
        # shuffle data
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        # do batch training
        for i in range(0, X_train.shape[0], batch_size):
            # get batch
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            # perform forward propagation
            h = forward(layers_, X_batch)

            # perform backwards propagation, updating weights
            if skip_first_layer:
                start = 1
            else:
                start = 0
            grad = layers_[-1].gradient(Y_batch, h)
            for layer in reversed(layers_[start:-1]):
                newGrad = layer.backward(grad)
                if (isinstance(layer, layers.FullyConnectedLayer)):
                    layer.updateWeights(grad, epoch, learning_rate)
                if (isinstance(layer, layers.Conv2DLayer)) or (isinstance(layer, layers.Conv3DLayer)):
                    layer.updateKernel(grad, epoch, learning_rate)
                grad = newGrad

        # evaluate loss for training
        h = forward(layers_, X_train)
        eval = layers_[-1].eval(Y_train, h)
        loss_train.append(eval)

        # finish training if change in loss is too small
        if (epoch > 2 and abs(eval - lastEval) < condition):
            break
        lastEval = eval

        # evaluate loss for validation
        h = forward(layers_, X_val)
        val_eval = layers_[-1].eval(Y_val, h)
        loss_val.append(val_eval)

        # pbar.set_description(f"Validation loss: {val_eval}")

        # print("Epoch: %d, Train Loss: %f, Val Loss: %f" % (epoch, eval, val_eval))
        epoch += 1
        pbar.update(1)

    calculate_accuracy(X_train, Y_train, layers_, type = "Training")
    calculate_accuracy(X_val, Y_val, layers_, type = "Validation")
    pbar.close()

    # plot log loss
    plt.xlabel("Epoch")
    plt.ylabel("J")
    plt.plot(loss_train, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.legend()
    plt.savefig(f'{filename}.png')
    plt.clf()

def calculate_accuracy(X, Y, layers, type = "Training"):
    Yhat = forward(layers, X)
    accuracy = (Y.argmax(axis=1) == Yhat.argmax(axis=1)).mean() * 100
    print(f"{type} accuracy: {accuracy}")