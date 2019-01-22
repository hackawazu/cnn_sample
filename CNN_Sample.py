#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Add, Input, Multiply, Concatenate

import os, glob, random
from PIL import Image
import pickle
import numpy as np
from sklearn import model_selection
import os


# In[2]:


batch_size = 10
num_classes = 2
epochs = 50
data_augmentation = True
num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = 'model0.h5'

img_width = 128
img_height = 128

#train_imgs = './Images/Train/'
#test_imgs = './Images/Test/'
data_imgs = './cnn_sample/Images/' # 適宜書き換え

class_label = ['class1', 'class2']


# In[3]:


def MyNet():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_height, img_width, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

# In[4]:
def load_images_all(dataset_path, shuffle=False):
    print('\tload images <-- {}'.format(dataset_path))
    if not os.path.exists(dataset_path):
        raise Exception('{} not exists'.format(dataset_path))

    cls_dirs = os.listdir(dataset_path)
    cls = 0
    imgs = []
    labels = []
    filepaths = []

    for cls_dir in cls_dirs:
        if not os.path.isdir(dataset_path + '/' + cls_dir): continue
        _imgs, _labels, _filepaths = load_images(dataset_path + '/' + cls_dir, cls)
        imgs += _imgs
        labels += _labels
        filepaths += _filepaths
        cls += 1

    imgs = np.array(imgs)
    labels = np.array(labels)
    filepaths = np.array(filepaths)

    if shuffle:
        s = np.arange(imgs.shape[0])
        np.random.shuffle(s)
        imgs = imgs[s]
        labels = labels[s]
        filepaths = filepaths[s]
    print('\tloaded images\n')
    return imgs, labels, filepaths

##  this is used in load_images_all
def load_images(dataset_path, label, shuffle=False):
    filepaths_jpg = glob.glob(dataset_path + '/*.jp*g')
    filepaths_png = glob.glob(dataset_path + '/*.png')
    filepaths = filepaths_jpg + filepaths_png
    filepaths.sort()
    datasets = []
    labels = []

    for filepath in filepaths:
        img = Image.open(filepath).convert('RGB') ## Gray->L, RGB->RGB
        img = img.resize((img_width, img_height))
        #label = int(filepath.split('/')[-1].split('_')[0])

        x = np.array(img, dtype=np.float32)
        x = x / 255.
        #x = x.reshape(3, INPUT_HEIGHT, INPUT_WIDTH)
        #t = np.array(label, dtype=np.int32)
        t = label
        datasets.append(x)
        labels.append(t)
    if shuffle: random.shuffle(datasets)

    return datasets, labels, filepaths


# In[5]:


def train():
    print('\ntrain start\n')

## load train and test images
    #x_train, y_train, filepaths_train = load_images_all(train_imgs, shuffle=True)
    #x_test, y_test, filepaths_test = load_images_all(test_imgs)
    x_data, y_data, filepaths_data = load_images_all(data_imgs)
    # 訓練データとテストデータに分ける
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        # 入力, 正解, 割合
        x_data, y_data, test_size=0.3
    )


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ## define model architecture
    model = Mynet()

    ## visualize model
    model.summary()

    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
    check_point = keras.callbacks.ModelCheckpoint(
        filepath = os.path.join('model0','model{epoch:02d}-vacc{val_acc:.2f}.hdf5'), 
        monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[early_stopping, check_point]
        )

    # Save model and weights
    #if not os.path.isdir(save_dir):
    #    os.makedirs(save_dir)
    #model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('\nSaved trained model at --> %s ' % model_path)
    print('\ntrain end\n')


# In[6]:


def test():
    print('\ntest start\n')
    model = MyNet()
    #model.summary()
    model.load_weights(model_path)

    x_test, y_test, paths = load_images_all(test_imgs)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

    # Evaluatemodel with test data set and share sample prediction results
    #evaluation = model.evaluate_generator((x_test, y_test),
    #                                  steps=x_test.shape[0] // batch_size)

    count = 0
    total = x_test.shape[0]
    print('---------------------------------------------------------------------')
    print('evaluate start')
    print('---------------------------------------------------------------------')
    for index, x in enumerate(x_test):
        x = x[None, ...]
        pred = model.predict(x, batch_size=1, verbose=0)
        score = np.max(pred)
        pred_label = np.argmax(pred)

        gt = y_test[index]

        if pred_label == gt: count += 1

        print(' {} : gt {} / pred {}'.format(paths[index], gt, pred_label))

    print('--------------------------------------------------------------------')
    print('accuracy {}  ({} / {})'.format(1.0*count/total, count, total))

    print('\ntest end\n')


# In[7]:


train()


# In[ ]:




