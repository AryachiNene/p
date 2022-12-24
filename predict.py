import models
import numpy as np
import os, copy
from PIL import Image

HEIGHT = 512  # 图像的长
WIDHT = 512  # 图像的宽
CLASS_NUMBERS = 2  # 分类数
class_colors = [[0, 0, 0], [0, 255, 0]]  # 黑，绿


def get_model():
    #加载之前构建的模型以及训练得到的权重参数
    model = models.main()
    model.load_weights('.\callbacks\last.h5')
    return model

#图像预处理
#改变大小、归一化
def precess_img(img):

    test_img = img.resize((HEIGHT, WIDHT))
    test_img_array = np.array(test_img)
    test_img_array = test_img_array / 255
    test_img_array = test_img_array.reshape(-1, HEIGHT, WIDHT, 3)
    return test_img_array

#预测
def predicting(model):
    test_data_path = "./dataset/testData/img"
    test_data = os.listdir(test_data_path)
    print(test_data)

    for test_name in test_data:

        test_img_full_path = os.path.join(test_data_path, test_name)
        test_img = Image.open(test_img_full_path)
        old_test_img = copy.deepcopy(test_img)
        test_img_array = np.array(test_img)
        original_height = test_img_array.shape[0]
        original_width = test_img_array.shape[1]
        test_img_array = precess_img(test_img)
        predict_picture = model.predict(test_img_array)[0]  #
        predict_picture = predict_picture.reshape((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))

        predict_picture = predict_picture.argmax(axis=-1)

        seg_img_array = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), 3))

        colors = class_colors

        for cn in range(CLASS_NUMBERS):

            seg_img_array[:, :, 0] += ((predict_picture[:, :] == cn) * colors[cn][0]).astype('uint8')
            seg_img_array[:, :, 1] += ((predict_picture[:, :] == cn) * colors[cn][1]).astype('uint8')
            seg_img_array[:, :, 2] + ((predict_picture[:, :] == cn) * colors[cn][2]).astype('uint8')
        seg_img = Image.fromarray(np.uint8(seg_img_array))

        seg_img = seg_img.resize((original_width, original_height))

        print(old_test_img.size)
        print(seg_img.size)
        image = Image.blend(old_test_img, seg_img, 0.3)

        save_path = './dataset/testData'
        save_path = os.path.join(save_path, 'imgout')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        image.save(save_path + '/' + test_name)



def main():
    model = get_model()
    predicting(model)

if __name__ == '__main__':
    main()
