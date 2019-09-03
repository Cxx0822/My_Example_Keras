from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from load_data import *


def my_predict():

    model = load_model("model.h5")

    test_generator = load_data(test_dir, 1)

    for test_image, test_labels in test_generator:
        prediction = model.predict(test_image) 
        max_index = np.argmax(prediction)  

        # 判断类别
        if max_index == 0:
            label = '%.2f%% ' % (prediction[0][0] * 100) + 'is a ' + str(my_labels[0]) + '.'
        elif max_index == 1:
            label = '%.2f%% ' % (prediction[0][1] * 100) + 'is a ' + str(my_labels[1]) + '.'
        elif max_index == 2:
            label = '%.2f%% ' % (prediction[0][2] * 100) + 'is a ' + str(my_labels[2]) + '.'
        elif max_index == 3:
            label = '%.2f%% ' % (prediction[0][3] * 100) + 'is a ' + str(my_labels[3]) + '.'
        elif max_index == 4:
            label = '%.2f%% ' % (prediction[0][4] * 100) + 'is a ' + str(my_labels[4]) + '.'

        plt.imshow(test_image[0])
        plt.title(label)
        plt.show()


if __name__ == "__main__":
    my_predict()
