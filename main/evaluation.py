import os
import numpy as np
import argparse
import pywt
from PIL import Image
from skimage.restoration import denoise_wavelet
from random import randint
from ISR.models import RDN

from resnet import resnet

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class testset:
    def __init__(self, path):
        test_data = np.load(os.path.join(path, "selected_test_data.npy"))
        self.test_data = test_data.astype(float)
        self.test_label = np.load(os.path.join(path, "selected_test_label.npy"))


def accuracy(pred, real):
    return np.sum(np.argmax(pred, 1) == np.argmax(real, 1)) / pred.shape[0]


def l2_distortion(img, adv):
    return np.sum(np.mean(np.sqrt((img - adv) ** 2), axis=0)) / (img.shape[1] * img.shape[2] * img.shape[3])


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


def wavelet_shrink(data):
    wavrec_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        wavelet_image = np.zeros((32, 32, 3))
        for j in range(3):
            im = Image.fromarray(np.uint8(data[i, :, :, j]))
            im = im.resize((64, 64), Image.BICUBIC)
            coeffs2 = pywt.dwt2(np.array(im), 'db4')
            LL, (LH, HL, HH) = coeffs2
            LL = LL[3:35, 3:35]
            LL = (LL - np.min(LL)) / (np.max(LL) - np.min(LL))
            im = Image.fromarray(np.uint8(LL * 255))
            wavelet_image[:, :, j] = np.array(im)
        wavrec_data[i] = wavelet_image

    return wavrec_data


def wavelet_LL(data):
    wavrec_data = np.zeros((data.shape[0], 16, 16, 3))
    for i in range(data.shape[0]):
        wavelet_image = np.zeros((16, 16, 3))
        for j in range(3):
            coeffs2 = pywt.dwt2(data[i, :, :, j], 'db4')
            LL, (LH, HL, HH) = coeffs2
            LL = LL[3:19, 3:19]
            LL = (LL - np.min(LL)) / (np.max(LL) - np.min(LL))
            im = Image.fromarray(np.uint8(LL * 255))
            wavelet_image[:, :, j] = np.array(im)
        wavrec_data[i] = wavelet_image

    return wavrec_data


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
        img = pixel_deflection(data[i], deflections=50, window=5)
        pd_data[i] = denoise_wavelet(img / 255.0, multichannel=True, convert2ycbcr=True,
                                     method='BayesShrink', mode='soft', sigma=0.04) * 255.0

    return pd_data


def super_resolution(data, rdn):
    sr_data = np.zeros((data.shape[0], 32, 32, 3))
    for i in range(data.shape[0]):
        sr_data[i] = rdn.predict(data[i])

    return sr_data


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-a", "--attack", help="Attack of adversarial examples", default="fgsm")
    # parser.add_argument("-e", "--eps", help="error tolerance", default=0.03)
    # parser.add_argument("-m", "--model", help="model type", default="clean")
    # args = parser.parse_args()

    data_path = "../data/data_original"
    data = testset(data_path)
    test_data, test_label = data.test_data, data.test_label
    # ae_path = "../results/model_clean_ae_original_mifgsm_0.005.npy"
    ae_path = "../results/bpda_base_ae.npy"
    ae_data = np.load(ae_path)

    print(ae_path)
    print("AE L2-distortion:", l2_distortion(test_data / 255.0, ae_data))

    base_model_path = "../saved_models/cifar10_resnet_clean.h5"

    n = 3
    depth = n * 9 + 2
    input_shape = test_data.shape[1:]
    base_model = resnet(input_shape=input_shape, depth=depth, num_classes=10).build()
    base_model.load_weights(base_model_path)

    base_pred = base_model.predict(test_data / 255.0)
    wd_pred = base_model.predict(wavelet_denoising(test_data))
    wavrec_data = wavelet_recovery(test_data) / 255.0
    wavrec_pred = base_model.predict(wavrec_data)
    wavshrink_data = wavelet_shrink(test_data) / 255.0
    wavshrink_pred = base_model.predict(wavshrink_data)

    '''
    average_accuracy = 0.0
    for _ in range(10):
        pd_pred = base_model.predict(pd_defense(test_data) / 255.0)
        average_accuracy += accuracy(pd_pred, test_label)
    average_accuracy /= 10.0
    '''
    pd_pred_labels = np.zeros(test_label.shape)
    for _ in range(10):
        pd_pred = base_model.predict(pd_defense(test_data) / 255.0)
        pd_pred_labels += pd_pred

    print("\n######################################")
    print("Clean image test accuracy: %0.4f" % accuracy(base_pred, test_label))
    print("Wavelet denoising test accuracy: %0.4f" % accuracy(wd_pred, test_label))
    print("Wavelet recovery test accuracy: %0.4f" % accuracy(wavrec_pred, test_label))
    print("Wavelet shrink test accuracy: %0.4f" % accuracy(wavshrink_pred, test_label))
    print("Pixel deflection test accuracy: %0.4f" % accuracy(pd_pred_labels, test_label))

    ae_pred = base_model.predict(ae_data)
    wa_ae = wavelet_denoising(ae_data * 255)
    wd_ae_pred = base_model.predict(wa_ae)
    wavrec_ae = wavelet_recovery(ae_data * 255) / 255.0
    wavrec_ae_pred = base_model.predict(wavrec_ae)
    wavshrink_ae = wavelet_shrink(ae_data * 255) / 255.0
    wavshrink_ae_pred = base_model.predict(wavshrink_ae)

    '''
    average_accuracy = 0.0
    for _ in range(10):
        pd_ae_pred = base_model.predict(pd_defense(ae_data * 255) / 255.0)
        average_accuracy += accuracy(pd_ae_pred, test_label)
    average_accuracy /= 10.0
    '''
    pd_pred_labels = np.zeros(test_label.shape)
    for _ in range(10):
        pd_pred = base_model.predict(pd_defense(ae_data * 255) / 255.0)
        pd_pred_labels += pd_pred

    print("\n######################################")
    print("AE test accuracy: %0.4f" % accuracy(ae_pred, test_label))
    print("Wavelet denoising AE test accuracy: %0.4f" % accuracy(wd_ae_pred, test_label))
    print("Wavelet recovery AE test accuracy: %0.4f" % accuracy(wavrec_ae_pred, test_label))
    print("Wavelet shrink AE test accuracy: %0.4f" % accuracy(wavshrink_ae_pred, test_label))
    print("Pixel deflection AE test accuracy: %0.4f" % accuracy(pd_pred_labels, test_label))

    '''
    new_model_path = "../saved_models/cifar10_resnet_new.h5"
    new_data = np.concatenate([wavshrink_data, wavrec_data], axis=3)
    input_shape = new_data.shape[1:]
    new_model = resnet(input_shape=input_shape, depth=depth, num_classes=10).build()
    new_model.load_weights(new_model_path)

    new_ae_data = np.concatenate([wavshrink_ae, wavrec_ae], axis=3)
    new_pred = new_model.predict(new_data)
    new_ae_pred = new_model.predict(new_ae_data)

    print("\n######################################")
    print("Shrink + recovery test accuracy: %0.4f" % accuracy(new_pred, test_label))
    print("Shrink + recovery AE test accuracy: %0.4f" % accuracy(new_ae_pred, test_label))

    sr_model_path = "../saved_models/cifar10_resnet_wavsr.h5"
    rdn = RDN(weights='psnr-small')
    LL = wavelet_LL(test_data)
    sr_data = super_resolution(LL, rdn)
    sr_pred = base_model.predict(sr_data / 255.0)
    sr_hybrid = np.concatenate([sr_data / 255.0, wavrec_data], axis=3)
    input_shape = sr_hybrid.shape[1:]
    sr_model = resnet(input_shape=input_shape, depth=depth, num_classes=10).build()
    sr_model.load_weights(sr_model_path)
    sr_hybrid_pred = sr_model.predict(sr_hybrid)

    LL = wavelet_LL(ae_data)
    sr_ae_data = super_resolution(LL, rdn)
    sr_hybrid_ae = np.concatenate([sr_ae_data / 255.0, wavrec_ae], axis=3)
    sr_hybrid_ae_pred = sr_model.predict(sr_hybrid_ae)

    print("\n######################################")
    print("SR test accuracy: %0.4f" % accuracy(sr_pred, test_label))
    print("SR hybrid test accuracy: %0.4f" % accuracy(sr_hybrid_pred, test_label))
    print("SR hybrid AE test accuracy: %0.4f" % accuracy(sr_hybrid_ae_pred, test_label))
    '''





