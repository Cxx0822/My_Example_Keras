from keras.preprocessing.image import ImageDataGenerator
import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

train_dir = my_data['train_dir']
valid_dir = my_data['valid_dir']
test_dir = my_data['test_dir']

BATCH_SIZE = my_data['batch_size']
IMG_SIZE = my_data['img_size'] 
my_labels = my_data['my_labels']


def load_data(data_dir, batch_size):
    # 数据集处理
    data_gen = ImageDataGenerator(rescale=1. / 255,)

    # 生成可迭代数据集对象
    data_generator = data_gen.flow_from_directory(data_dir, target_size=(IMG_SIZE, IMG_SIZE), 
                                                  batch_size=batch_size, class_mode='categorical')

    return data_generator


if __name__ == "__main__":
    data_generator = load_data(train_dir, BATCH_SIZE)

    import matplotlib.pyplot as plt
    import numpy as np
    # 测试数据集读取
    for batch_image, batch_labels in data_generator:
        print("Batch_image_shape: ", batch_image.shape)
        print("Batch_labels_shape: ", batch_labels.shape)
        label = np.argmax(batch_labels)  
        plt.imshow(batch_image[0])
        plt.title(my_labels[label])
        plt.show()
