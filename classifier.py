import tensorflow as tf
import tensorflow.keras as keras
import pickle as pkl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
from TrojanNet import TrojanNet

def split_and_save(output_path, name):
    images_path = "dataset"
    # load data, then split into train and test sets
    X_train, X_test, y_train, y_test, z, X, y = _prepare_img_data(images_path, 0.2)
    
    output = os.path.join(output_path, name)
    with open(output, "wb") as f:
        pkl.dump([X_train, y_train, X_test, y_test, z, X, y], f)


def train(reduceLROnPlat, validation_data, train_data, epochs=300, verbose=1, summary=True):
    X_train, y_train = train_data
    input_shape = X_train[0].shape
    model = TrojanNet(input_shape, num_labels=2)
    
    if summary:
        model.summary()
        
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
    
    hist = model.fit(
        X_train,y_train,
        validation_data=validation_data,
        epochs=epochs,
        verbose=verbose,
        callbacks=[reduceLROnPlat])

    return model, hist


def run_test(data_name, epochs=300, colab_folder=None, summary=False, verbose=1):
    X_train, X_test, y_train, y_test, z, X, y = load_data(data_name, colab_folder)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    reduceLROnPlat = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.95,
        patience=3,
        verbose=verbose,
        mode='min',
        min_delta=0.0001,
        cooldown=2,
        min_lr=0.000001
    )

    model, hist = train(reduceLROnPlat=reduceLROnPlat, 
                    validation_data=(X_test, y_test),
                    train_data=(X_train, y_train),
                    epochs=epochs,
                    verbose=verbose,
                    summary=summary)
    return model, hist, z, X_test, y_test


def _prepare_img_data(path, test_size):
    X, y, z = _load(path)
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test, z, X, y

def _load(path):
    emoji_types = os.listdir(path)
    emoji_types.remove('.DS_Store')
    print(emoji_types)
    
    X = [] # images
    y = [] # labels

    for emoji_type in emoji_types:
        emoji_type_path = path + "/" + emoji_type

        class_num = emoji_types.index(emoji_type)

        for name in os.listdir(emoji_type_path):
            if name == ".DS_Store":
                continue
            
            img_array = cv2.imread(os.path.join(emoji_type_path, name))

            X.append(img_array)
            y.append(class_num)

    return np.array(X), np.array(y), np.array(emoji_types)


def load_data(name, colab_folder):
    path = name + ".pkl"

    if colab_folder:
        path = colab_folder + path
    
    print(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(path + " is not found, use split_and_save function in utils.py to save data.")

    with open(path, 'rb') as handle:
        X_train, y_train, X_test, y_test, z, X, y = pkl.load(handle)
        return X_train, X_test, y_train, y_test, z, X, y
    
def plot(history, path=None, name=None, save=False):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, name))

    plt.show()
    plt.close()

def draw_confusion_matrix(X, y_true, model, labels, path=None, name=None, save=False):
    fig = plt.figure(figsize=(10,7))
    y_pred = model.predict(X)

    y_true = [ labels[n] for n in np.argmax(y_true, 1)]
    y_pred = [ labels[n] for n in np.argmax(y_pred, 1)]

    matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(matrix, columns=np.unique(y_true), index = np.unique(y_true))

    sns.heatmap(df_cm,  annot=True, square=True, fmt='.0f', cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values');

    if save:
        plt.savefig(os.path.join(path, name))

    plt.show()
    plt.close()
  
data_path = "saved_dataset/"
if not os.path.exists(data_path):
        os.mkdir(data_path)
split_and_save(data_path, "dat.pkl")

data_name = 'dat'
epochs = 2
DATA_PATH = './saved_dataset/'
model, hist, z, X_test, y_test = run_test(data_name=data_name,
                                          colab_folder=DATA_PATH,
                                          epochs=epochs,
                                          summary=True,
                                          verbose=1)

plot(hist, path="./", name=None, save=False)
draw_confusion_matrix(X=X_test, y_true=y_test, model=model, labels=z, path="./", name=None, save=False)