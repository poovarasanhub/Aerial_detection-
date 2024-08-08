if True:
    from reset_random import reset_random
    reset_random()
import glob
import os

import cmapy
import numpy as np
import tqdm
from cv2 import cv2
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.densenet import preprocess_input as de_pp
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as iv_pp
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image

from utils import DATASET


def get_densenet_201():
    print('[INFO] Building DenseNet201 Model')
    model = DenseNet201(include_top=False,
                        weights='weights/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
    return model


def get_inception_v3():
    print('[INFO] Building InceptionV3 Model')
    model = InceptionV3(include_top=False,
                        weights='weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    return model


def get_feature_map_model(model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    feature_map_model = Model(model.input, layer_outputs)
    return feature_map_model


def get_image_to_predict(im_path, size, pp):
    img = image.load_img(im_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return pp(x)


def get_feature(img, model):
    feature = model.predict(img)
    feat = feature.flatten()
    return list(feat)


def get_feature_image(img, model):
    feature_map = model.predict(img)[-1]
    feature_image = feature_map[0, :, :, -1]
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std()
    feature_image *= 64
    feature_image += 128
    feature_image = np.clip(feature_image, 0, 255).astype('uint8')
    return feature_image


d201_shape = (224, 224, 3)
iv3_shape = (299, 299, 3)

d201 = get_densenet_201()
iv3 = get_inception_v3()

d201_fmm = get_feature_map_model(d201)
iv3_fmm = get_feature_map_model(iv3)

if __name__ == '__main__':
    from reset_random import reset_random

    reset_random()
    data_dir = 'Data/preprocessed'
    save_dir = 'Data/fused_features'

    for d in DATASET:
        features = []
        labels = []
        print('[INFO] Feature Extraction Dataset :: {0}'.format(d))
        classes = DATASET[d]
        for c in classes:
            images_list = sorted(glob.glob(os.path.join(data_dir, d, c, '*.jpg')))
            save_path = os.path.join(save_dir, d, c)
            os.makedirs(save_path, exist_ok=True)
            for img_path in tqdm.tqdm(images_list, desc='[INFO] {0} :'.format(c)):
                im_save_path = os.path.join(save_path, os.path.basename(img_path))
                d201_im = get_image_to_predict(img_path, d201_shape, de_pp)
                iv3_im = get_image_to_predict(img_path, iv3_shape, iv_pp)
                if not os.path.isfile(im_save_path):
                    r101_fm = get_feature_image(d201_im, d201_fmm)
                    iv3_fm = get_feature_image(iv3_im, iv3_fmm)

                    r101_fm = cv2.resize(r101_fm, (48, 48))
                    iv3_fm = cv2.resize(iv3_fm, (48, 48))

                    r101_fm = cv2.applyColorMap(r101_fm, cmapy.cmap('viridis_r'))
                    iv3_fm = cv2.applyColorMap(iv3_fm, cmapy.cmap('viridis_r'))

                    im_save_path = os.path.join(save_path, os.path.basename(img_path))

                    cv2.imwrite(im_save_path, r101_fm + iv3_fm)
                else:
                    d201_fe = get_feature(d201_im, d201)[:224]
                    iv3_fe = get_feature(iv3_im, iv3)[:299]
                    d201_iv3_fe = [*d201_fe, *iv3_fe]

                    features.append(d201_iv3_fe)
                    labels.append(classes.index(c))

        print('[INFO] Saving Features and Labels')
        f_path = os.path.join(save_dir, d, 'features.npy')
        features = np.array(features, ndmin=2)
        np.save(f_path, features)
        labels = np.array(labels)
        l_path = os.path.join(save_dir, d, 'labels.npy')
        np.save(l_path, labels)
