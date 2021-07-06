import os
import numpy as np
import argparse
import keras.backend as K
import tensorflow as tf
from resnet import resnet
import warnings
import pywt
from PIL import Image
from random import randint
from skimage.restoration import denoise_wavelet

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class testset:
    def __init__(self, path):
        test_data = np.load(os.path.join(path, "selected_test_data.npy"))
        self.test_data = test_data.astype(float)
        self.test_label = np.load(os.path.join(path, "selected_test_label.npy"))


def accuracy(pred, real):
    return np.sum(np.argmax(pred, 1) == np.argmax(real, 1)) / pred.shape[0]


def pixel_deflection(img, deflections, window):
    H, W, C = img.shape
    while deflections > 0:
        for c in range(C):
            x, y = randint(0, H-1), randint(0, W-1)

            while True: #this is to ensure that PD pixel lies inside the image
                a, b = randint(-1*window, window), randint(-1*window, window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            img[x, y, c] = img[x+a, y+b, c]
            deflections -= 1
    return img


def pd_defense(data):
    pd_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = pixel_deflection(data[i], deflections=30, window=5)
        pd_data[i] = denoise_wavelet(img / 255.0, multichannel=True, convert2ycbcr=True,
                                     method='BayesShrink', mode='soft', sigma=0.04) * 255.0

    return pd_data


def wavelet_denoising(data):
    wd_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        wd_data[i] = denoise_wavelet(data[i] / 255.0, multichannel=True, convert2ycbcr=True,
                                     method='BayesShrink', mode='soft', rescale_sigma=False)

    return wd_data


def wavelet_recovery(data):
    wavrec_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        wavelet_image = np.zeros((32, 32, 3))
        for j in range(3):
            coeffs2 = pywt.dwt2(data[i, :, :, j], 'db4')
            LL, (LH, HL, HH) = coeffs2
            LL = LL[3:19, 3:19]
            LL = (LL - np.min(LL)) / (np.max(LL) - np.min(LL))
            im = Image.fromarray(np.uint8(LL * 255))
            im = im.resize((32, 32), Image.BICUBIC)
            wavelet_image[:, :, j] = np.array(im)
        wavrec_data[i] = wavelet_image

    return wavrec_data


def get_random_targets(label):
    result = np.zeros(label.shape[0])
    nb_s = label.shape[0]
    nb_classes = label.shape[1]

    for i in range(nb_s):
        result[i] = np.argmax(np.roll(label[i, :], randint(1, nb_classes - 1)))

    return result.astype(np.int64)


def pd_attack(model, adv, label):

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    out = tf.log(model(x / 255.0))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=label)
    grads = tf.gradients(loss, [x])[0]
    grads = tf.sign(grads)

    sess = K.get_session()

    initial = adv.copy()
    img = adv.copy()
    for i in range(5):
        # print('def pred', np.argmax(sess.run(out, {x: pd_adv})), sess.run(loss, {x: pd_adv}))
        res = pd_defense(img)
        img -= sess.run(grads, {x: res})
        img = np.clip(img, initial - 1, initial + 1)
        img = np.clip(img, 0, 255)

    return img


def apply_pd_attack(model, img, label):
    pd_ae = np.zeros(img.shape)
    for i in range(img.shape[0]):
        pd_ae[i] = pd_attack(model, img[i], label[i])

    return pd_ae


def l2_distortion(img, adv):
    l2 = 0.0
    for i in range(img.shape[0]):
        l2 += np.sqrt(np.sum((img[i] - adv[i]) ** 2))
    return l2 / img.shape[0]


def linf_distortion(img, adv):
    linf = 0.0
    for i in range(img.shape[0]):
        linf += np.max(np.abs(img[i] - adv[i]))
    return linf / img.shape[0]


def bpda_base_attack(model, adv, label):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    out = tf.log(model(x / 255.0))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=label)
    # labels=get_random_targets(label))
    grads = tf.gradients(loss, [x])[0]
    grads = tf.sign(grads)

    sess = K.get_session()

    initial = adv.copy()
    img = adv.copy()
    for i in range(5):
        # print('def pred', np.argmax(sess.run(out, {x: pd_adv})), sess.run(loss, {x: pd_adv}))
        res = img
        img -= sess.run(grads, {x: res})[:, :, :, :3]
        img = np.clip(img, initial - 4, initial + 4)
        img = np.clip(img, 0, 255)

    return img


def bpda_attack(model, adv, label):
    x = tf.placeholder(tf.float32, [None, 32, 32, 6])
    out = tf.log(model(x / 255.0))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=label)
    # labels=get_random_targets(label))
    grads = tf.gradients(loss, [x])[0]
    grads = tf.sign(grads)

    sess = K.get_session()

    initial = adv.copy()
    img = adv.copy()
    for i in range(5):
        # print('def pred', np.argmax(sess.run(out, {x: pd_adv})), sess.run(loss, {x: pd_adv}))
        res = hybrid_defense(img)
        img -= sess.run(grads, {x: res})[:, :, :, :3]
        img = np.clip(img, initial - 1, initial + 1)
        img = np.clip(img, 0, 255)

    return img


def hybrid_defense(img):
    return np.concatenate([wavelet_denoising(img) * 255, wavelet_recovery(img)], axis=3)


if __name__ == '__main__':
    data_path = "../data/data_original"
    data = testset(data_path)
    test_data, test_label = data.test_data, data.test_label

    dist = "4"
    itern = "i5"
    base_model_path = "../saved_models/cifar10_resnet_clean.h5"

    n = 3
    depth = n * 9 + 2

    input_shape = test_data.shape[1:]
    base_model = resnet(input_shape=input_shape, depth=depth, num_classes=10).build()
    base_model.load_weights(base_model_path)

    base_pred = base_model.predict(test_data / 255.0)
    average_accuracy = 0.0
    for _ in range(10):
        pd_pred = base_model.predict(pd_defense(test_data) / 255.0)
        average_accuracy += accuracy(pd_pred, test_label)
    average_accuracy /= 10.0

    print("\n######################################")
    print("Clean image test accuracy: %0.4f" % accuracy(base_pred, test_label))
    print("Clean image PD test accuracy: %0.4f" % average_accuracy)

    random_label = get_random_targets(test_label)

    bpda_base_ae = bpda_base_attack(base_model, test_data, random_label)
    np.save("../results/bpda_base_ae_{}_{}.npy".format(dist, itern), bpda_base_ae / 255.0)

    bpda_base_ae_pred = base_model.predict(bpda_base_ae / 255.0)
    print("\n######################################")
    print("BPDA base AE test accuracy: %0.4f" % accuracy(bpda_base_ae_pred, test_label))

    pd_bpda_ae = pd_attack(base_model, test_data, random_label)
    np.save("../results/pd_bpda_ae_{}_{}.npy".format(dist, itern), pd_bpda_ae / 255.0)

    average_accuracy = 0.0
    for _ in range(10):
        pd_ae_pred = base_model.predict(pd_defense(pd_bpda_ae) / 255.0)
        average_accuracy += accuracy(pd_ae_pred, test_label)
    average_accuracy /= 10.0

    print("\n######################################")
    print("PD BPDA AE test accuracy: %0.4f" % average_accuracy)
    print("AE L2-distortion:", l2_distortion(test_data / 255.0, pd_bpda_ae / 255.0))
    print("AE Linf-distortion:", linf_distortion(test_data / 255.0, pd_bpda_ae / 255.0))

    hybrid_model_path = "../saved_models/cifar10_resnet_hybrid.h5"
    input_shape = (32, 32, 6)
    hybrid_model = resnet(input_shape=input_shape, depth=depth, num_classes=10).build()
    hybrid_model.load_weights(hybrid_model_path)

    bpda_hybrid_ae = bpda_attack(hybrid_model, test_data, random_label)
    bpda_hybrid_ae_pred = hybrid_model.predict(hybrid_defense(bpda_hybrid_ae) / 255.0)

    np.save("../results/hybrid_bpda_ae_{}_{}.npy".format(dist, itern), bpda_hybrid_ae / 255.0)
    print("\n######################################")
    print("Hybrid BPDA AE test accuracy: %0.4f" % accuracy(bpda_hybrid_ae_pred, test_label))
    print("AE L2-distortion:", l2_distortion(test_data / 255.0, bpda_hybrid_ae / 255.0))
    print("AE Linf-distortion:", linf_distortion(test_data / 255.0, bpda_hybrid_ae / 255.0))
