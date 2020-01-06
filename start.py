from helpers.params_helper import ParamsHelper
from classificators.vgg16_classificator import VGG16Classificator
from classificators.base_classificator import BaseClassificator

from os import listdir
from os.path import join

if __name__ == '__main__':
    train_data_dir = 'datasets/v_data/train'
    validation_data_dir = 'datasets/v_data/test'
    params_helper = ParamsHelper()

    params = {
        'train_data_dir': train_data_dir, 
        'validation_data_dir': validation_data_dir,
        'number_train_samples': params_helper.calc_number_samples(train_data_dir),
        'number_validation_samples': params_helper.calc_number_samples(validation_data_dir),
        'epochs': 10,
        'batch_size': 8
    }

    classificator = VGG16Classificator(params)

    classificator.learn()

    # classificator.load_model()
    # predictions = classificator.predict_images_from_folder('datasets/v_data/test', ['planes'], 10) 
    # print(predictions)   
