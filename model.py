import sys
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint


def get_record_and_image(index):
    return record, Image.open(path)


def get_image_training_data(path):
    image = Image.open(path)
    # Scale the image down to half resolution
    image = image.resize((image.width // 4, image.height // 4), Image.BILINEAR)
    image_data = np.array(image)
    # Return image data  normalized to range [-0.5, 0.5]
    return image_data / 255.0 - 0.5


def read_driving_data(data_folder, include_side_images=False):
    log_path = os.path.join(data_folder, 'driving_log.csv')
    df = pd.read_csv(log_path)
    for i in tqdm(range(len(df))):
        record = df.iloc[i]
        steering_angle = record['steering']
        steering_angle *= 0.15
        image_path = os.path.join(data_folder, record.center.strip())
        image_data = get_image_training_data(image_path)
        yield image_data, steering_angle
        if include_side_images:
            for path in [record.left, record.right]:
                image_path = os.path.join(data_folder, path.strip())
                image_data = get_image_training_data(image_path)
                yield image_data, steering_angle * 1.5

            
def create_training_data(data_folder, include_side_images=False):
    print('Creating training data from {}'.format(data_folder))
    X_train = []
    y_train = []
    for image_data, steering_angle in read_driving_data(data_folder, include_side_images):
        # if random.random() < 0.5:
        #     image_data = np.fliplr(image_data)
        #     steering_angle = -steering_angle
        X_train.append(image_data)
        y_train.append(steering_angle)
    print('{} total training samples'.format(len(X_train)))
    return np.array(X_train), np.array(y_train)


def show_layer_info(model):
    print('Layer info for model:')
    for n, layer in enumerate(model.layers, 1):
        print('  Layer {:2} {:16} input shape {} output shape {}'.format(n, layer.name, layer.input_shape, layer.output_shape))


# def train(model, training_generator, samples_per_epoch, nb_epoch=10):
#     model.compile('adam', 'mse')
#     model.fit_generator(training_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=2)
#     model.save('model.h5')


def train(model, X_train, y_train, nb_epoch=10):
    model.compile('adam', 'mse')
    callbacks = [ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(X_train, y_train, validation_split=0.2, nb_epoch=nb_epoch, verbose=2, callbacks=callbacks)
    model.save('model.h5')


def create_model():
    model = Sequential()
    #model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), input_shape=(80, 160, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(40, 80, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model


# def generator(X_train, y_train):
#     assert len(X_train) == len(y_train)
#     sample_count = len(X_train)
#     while 1:
#         #i = randrange(sample_count)
#         # for i in range(sample_count):
#         #     yield X_train[i:i+1], y_train[i:i+1]
#         yield X_train, y_train

nb_epochs = int(sys.argv[1])

X_list = []
y_list = []
for data_folder in sys.argv[2:]:
    X, y = create_training_data(data_folder, False)
    X_list.append(X)
    y_list.append(y)
X_train = np.concatenate(X_list)
y_train = np.concatenate(y_list)
    

model = create_model()
show_layer_info(model)
#training_generator = generator(X_train, y_train)

train(model, X_train, y_train, nb_epochs)

