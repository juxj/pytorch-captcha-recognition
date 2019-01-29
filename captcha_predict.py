# -*- coding: UTF-8 -*-
from os import listdir

import numpy as np
import torch
from torch.autograd import Variable

# from visdom import Visdom # pip install Visdom
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN


def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()

    # vis = Visdom()

    results = []
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)
        results.append(c)

        # vis.images(image, opts=dict(caption=c))
    codes = get_file_codes()
    hits = get_result(codes, results)

    print float(len(hits)) / float(len(codes)) * 100


def get_result(codes, results):
    hits = []
    for code in codes:
        for item in results:
            if code == item:
                hits.append(code)
    return hits


def get_file_codes():
    codes = []
    files = listdir(captcha_setting.PREDICT_DATASET_PATH)
    for item in files:
        code = item.split("_")[0]
        codes.append(code)
    return codes


if __name__ == '__main__':
    main()
