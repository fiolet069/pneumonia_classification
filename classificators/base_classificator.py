from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import cv2
import numpy


class BaseClassificator():    
    def __init__(self, params):
        self.__img_width, self.__img_height = 512, 512
        self.__train_data_dir = params['train_data_dir']
        self.__validation_data_dir = params['validation_data_dir']
        self.__number_train_samples = params['number_train_samples']
        self.__number_validation_samples = params['number_validation_samples']
        self.__epochs = params['epochs']
        self.__batch_size = params['batch_size']
        self.__define_input_shape()

        self.__name_saved_model = 'saved_models/base_classificator.h5'
        self.__name_load_model = 'saved_models/base_classificator.h5'


    def __define_input_shape(self):
        if K.image_data_format() == 'channels_first':
            self.__input_shape = (3, self.__img_width, self.__img_height)
        else:
            self.__input_shape = (self.__img_width, self.__img_height, 3)


    def __define_generators(self):        
        self.__train_datagen = ImageDataGenerator(
            rescale = 1. / 255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
        )

        self.__test_datagen = ImageDataGenerator(rescale = 1. / 255)

        self.__train_generator = self.__train_datagen.flow_from_directory(
            self.__train_data_dir,
            target_size = (self.__img_width, self.__img_height),
            batch_size = self.__batch_size,
            class_mode = 'binary'
        )

        self.__validation_generator = self.__test_datagen.flow_from_directory(
            self.__validation_data_dir,
            target_size = (self.__img_width, self.__img_height),
            batch_size = self.__batch_size,
            class_mode = 'binary'
        )
    

    # Override
    def __build_model(self):
        self.__model = Sequential()
        
        self.__model.add(Conv2D(32, (2, 2), input_shape = self.__input_shape)) 
        self.__model.add(Activation('relu')) 
        self.__model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        self.__model.add(Conv2D(32, (2, 2))) 
        self.__model.add(Activation('relu')) 
        self.__model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        self.__model.add(Conv2D(64, (2, 2))) 
        self.__model.add(Activation('relu')) 
        self.__model.add(MaxPooling2D(pool_size =(2, 2))) 
        
        self.__model.add(Flatten()) 
        self.__model.add(Dense(64)) 
        self.__model.add(Activation('relu')) 
        self.__model.add(Dropout(0.5)) 
        self.__model.add(Dense(1)) 
        self.__model.add(Activation('sigmoid'))        

        self.__model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


    def learn(self):
        self.__define_generators()
        self.__build_model()

        self.__model.fit_generator(
            self.__train_generator,
            steps_per_epoch = self.__number_train_samples,
            epochs = self.__epochs,
            validation_data = self.__validation_generator,
            validation_steps = self.__number_validation_samples
        )
        self.__model.save_weights(self.__name_saved_model)

    
    def load_model(self):
        self.__build_model()
        self.__model.load_weights(self.__name_load_model)


    # This method for testing
    def predict_images_from_folder(self, path_to_images, classes, number_samples):
        datagen = ImageDataGenerator(rescale = 1. / 255)
        generator = datagen.flow_from_directory(
            path_to_images,
            target_size = (self.__img_width, self.__img_height),
            batch_size = 1,
            class_mode = 'categorical',
            classes = classes
        )

        predictions = self.__model.predict_generator(generator, steps = number_samples)
        return predictions
