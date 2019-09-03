import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

IMG_SIZE = my_data['img_size'] 
N_CLASSES = my_data['n_classes']


def create_model():
    model = Sequential()

    # block 1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # block 3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # block 4
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # block 5
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=N_CLASSES, activation='softmax'))

    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()

    # draw the model structure  
    # pip install pydot, pip install graphviz
    # windows: https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi, 并添加环境变量
    from keras.utils.vis_utils import plot_model
    plot_model(model, show_shapes=True, to_file='model.png')
