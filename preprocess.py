import glob
import os
import random

import tqdm
from cv2 import cv2

from utils import DATASET


def contrast_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    lab_planes[0] = clahe_.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gaussian_filter(img):
    kernel_size = (5, 5)
    sigma_x = 1
    return cv2.GaussianBlur(img, kernel_size, sigma_x)


def augment(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


if __name__ == '__main__':
    data_dir = 'Data/source'
    save_dir = 'Data/preprocessed'
    for d in DATASET:
        print('[INFO] Preprocessing Dataset :: {0}'.format(d))
        classes = DATASET[d]
        for c in classes:
            if d == 'UCM':
                c = str(classes.index(c)).zfill(2)
            images_list = sorted(glob.glob(os.path.join(data_dir, d, c, '*.jpg' if d == 'AID' else '*.tif')))
            save_path = os.path.join(save_dir, d, classes[int(c)] if d == 'UCM' else c)
            os.makedirs(save_path, exist_ok=True)
            for img_path in tqdm.tqdm(images_list, desc='[INFO] {0} :'.format(classes[int(c)] if d == 'UCM' else c)):
                if d == 'AID':
                    image = cv2.imread(img_path)
                else:
                    image = cv2.VideoCapture(img_path).read()[1]
                im_save_path = os.path.join(save_path, os.path.basename(img_path)[:-3] + 'jpg')
                if not os.path.isfile(im_save_path):
                    gf = gaussian_filter(image)
                    ce = contrast_enhance(gf)
                    cv2.imwrite(im_save_path, ce)
                    aug_img = augment(ce, 30)
                    cv2.imwrite(im_save_path[:-4]+'_1.jpg', aug_img)
