from helpers.params_helper import ParamsHelper
from classificators.base_classificator import BaseClassificator

from os import listdir
from os.path import join

if __name__ == '__main__':
    train_data_dir = 'datasets/chest_xray/train'
    validation_data_dir = 'datasets/chest_xray/test'
    params_helper = ParamsHelper()

    params = {
        'size_image': (512, 512),
        'train_data_dir': train_data_dir, 
        'validation_data_dir': validation_data_dir,
        'number_train_samples': params_helper.calc_number_samples(train_data_dir),
        'number_validation_samples': params_helper.calc_number_samples(validation_data_dir),
        'epochs': 3,
        'batch_size': 1
    }

    classificator = BaseClassificator(params)

    classificator.learn()

    # classificator.load_model()
    # predictions = classificator.predict_images_from_folder('datasets/v_data/test', ['planes'], 10) 
    # print(predictions)   
