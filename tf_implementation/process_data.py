import glob
import cv2
import numpy as np
class data_set:
    def __init__(self, training_path, style_image_path, test_image_dir, test_image_path):
        self.training_path = training_path
        self.style_image_path = style_image_path
        self.test_image_dir = test_image_dir
        self.test_image_path = test_image_path
        self.training_file_paths = self.get_jpg_paths(training_path)

        self.style_image = np.expand_dims(self.get_image(style_image_path), axis=0)


    # get all jpg files from a folder
    def get_jpg_paths(self, folder):
        paths = glob.glob(folder + '/*.jpg')
        return paths

    # get a numpy array of (256,256,3) from an image path
    def get_image(self, filepath):
        img = cv2.imread(filepath)
        newimg = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        return newimg

    # get training image for a batch, return should be dimension of (batch, 256, 256, 3)
    def get_training_image(self, iteration, batch_size):
        start_index = (batch_size * iteration) % len(self.training_file_paths)
        X = []
        for i in xrange(0, batch_size):
            image = self.get_image(self.training_file_paths[start_index + i])
            X.append(image)
        return np.array(X)

#
# dataset = dataset('./training_data', './style_data/style.jpg', './test_data/', './test/data/baby.jpg')
# dataset.style_image
