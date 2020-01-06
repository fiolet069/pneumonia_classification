from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from classificators.base_classificator import BaseClassificator


class OrdinaryCnnClassificator(BaseClassificator):
    def __init__(self, params):
        BaseClassificator.__init__(self, params)

        BaseClassificator.__name_saved_model = 'saved_models/ordinary_cnn_classificator.h5'
        BaseClassificator.__name_load_model = 'saved_models/ordinary_cnn_classificator.h5'


    def __build_model(self):
        self.__model = VGG16()
        
        self.__model.add(Dense(1)) 
        self.__model.add(Activation('sigmoid')) 

        self.__model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    