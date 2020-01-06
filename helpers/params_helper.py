from os import listdir
from os.path import join


class ParamsHelper:
    def calc_number_samples(self, path_images):
        number = 0

        for folder in listdir(path_images):
            folder_path = join(path_images, folder)
            number += len(listdir(folder_path))

        return number
