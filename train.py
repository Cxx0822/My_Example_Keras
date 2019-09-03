from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD

import os

from model import *
from load_data import *
import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

train_dir = my_data['train_dir']
valid_dir = my_data['valid_dir']
test_dir = my_data['test_dir']

N_CLASSES = my_data['n_classes']
BATCH_SIZE = my_data['batch_size']
IMG_SIZE = my_data['img_size'] 

my_labels = my_data['my_labels']
max_step = my_data['max_step']


def my_train():
    # 加载数据集
    train_generator = load_data(train_dir, BATCH_SIZE)
    validation_generator = load_data(valid_dir, BATCH_SIZE)

    # 加载模型
    model = create_model()

    # 选择损失、优化器
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 训练
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=max_step,
                                  validation_data=validation_generator,
                                  validation_steps=50)

    model.save('model.h5')

if __name__ == "__main__":
    my_train()
