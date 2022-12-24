import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import models

CLASS_NUMBERS = 2
HEIGHT = 512
WIDHT = 512
batch_size = 4

#交叉熵损失函数
def customied_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred)
    return loss

#加载模型及参数
def get_model():
    model = models.main()
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    weights_path = get_file(filename, WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

    model.compile(loss=customied_loss, optimizer=Adam(1e-3), metrics=['accuracy'])

    return model

#获取训练样本和标签
def get_data():
    with open('./dataset/trainData/train.txt', 'r') as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    return lines, num_train, num_val

def set_callbacks():

    logdir = os.path.join("callbacks")
    print(logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        ModelCheckpoint(output_model_file, save_best_only=True, save_freq='epoch'),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-3, patience=10)#间隔一段时间保存最优权重参数
    ]

    return callbacks, logdir

def generate_arrays_from_file(lines, batch_size):

    numbers = len(lines)

    read_line = 0
    while True:

        x_train = []
        y_train = []

        for t in range(batch_size):

            if read_line == 0:
                np.random.shuffle((lines))

            train_x_name = lines[read_line].split(';')[0]

            img = Image.open('./dataset/trainData/jpg' + '/' + train_x_name)
            img = img.resize((WIDHT, HEIGHT))
            img_array = np.array(img)

            img_array = img_array / 255

            x_train.append(img_array)
            train_y_name = lines[read_line].split(";")[1].replace('\n', '')

            img = Image.open('./dataset/trainData/png' + '/' + train_y_name)
            img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))
            img_array = np.array(img)

            labels = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))

            for cn in range(CLASS_NUMBERS):
                labels[:, :, cn] = (img_array[:, :, 0] == cn).astype(int)
            labels = np.reshape(labels, (-1, CLASS_NUMBERS))
            y_train.append(labels)
            read_line = (read_line + 1) % numbers

        yield (np.array(x_train), np.array(y_train))


def main():
    model = get_model()

    lines, train_nums, val_nums = get_data()

    callbacks, logdir = set_callbacks()

    generate_arrays_from_file(lines, batch_size=4)

    model.fit_generator(generate_arrays_from_file(lines[:train_nums], batch_size),
                        steps_per_epoch=max(1, train_nums // batch_size),
                        epochs=50, callbacks=callbacks,
                        validation_data=generate_arrays_from_file(lines[train_nums:], batch_size),
                        validation_steps=max(1, val_nums // batch_size),
                        initial_epoch=0)

    save_weight_path = os.path.join(logdir,'last.h5')

    model.save_weights(save_weight_path)


if __name__ == '__main__':
    main()
