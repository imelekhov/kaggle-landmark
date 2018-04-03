import numpy as np
import cv2
import pandas as pd
import random
from io import BytesIO
from skimage.restoration import denoise_wavelet
cv2.ocl.setUseOpenCL(False)



def test_set_transform(img):
    
    if random.random() > 0.5:
        scales = [lambda x: augment_scale(x, 0.5, cv2.INTER_CUBIC),
                  lambda x: augment_scale(x, 0.8, cv2.INTER_CUBIC),
                  lambda x: augment_scale(x, 1.5, cv2.INTER_CUBIC),
                  lambda x: augment_scale(x, 2, cv2.INTER_CUBIC)
                  ]

        qual = [lambda x: augment_jpeg_compress(x, 70),
                lambda x: augment_jpeg_compress(x, 90)
                ]

        g = [lambda x: augment_random_gamma(x, 0.8),
             lambda x: augment_random_gamma(x, 1.2)
             ]


        transformations = scales+ qual+g
        transform_id = np.random.choice(len(transformations))
        trf = transformations[transform_id]

        img = trf(img)
    
    return img




def residual_image_noise(img, levels=1, wavelet='db1'):
    denoised = denoise_wavelet(img, multichannel=True, wavelet=wavelet, wavelet_levels=levels)*255
    noise = img-1.*denoised
    
    noise[:, :, 0] -= noise[:, :, 0].min()
    noise[:, :, 1] -= noise[:, :, 1].min()        
    noise[:, :, 2] -= noise[:, :, 2].min()
    
    noise[:, :, 0] /= noise[:, :, 0].max()
    noise[:, :, 1] /= noise[:, :, 1].max()        
    noise[:, :, 2] /= noise[:, :, 2].max()
    
    noise = np.round(noise*255).astype(np.uint8)
    
    return noise


def center_crop(img, size):

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]
    
    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w)/2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h)/2)
    img_pad = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=[0,0,0])
    w, h = img_pad.shape[1::-1]

    x1 = w//2-size[0]//2
    y1 = h//2-size[1]//2

    img_pad = img_pad[y1:y1+size[1], x1:x1+size[0], :]

    return img_pad


def augment_random_flip(img, hprob=0.5):
    img = img.copy()
    if random.random() > hprob:
        img = cv2.flip(img, 1)
        
    return img

def adjust_gamma(img, gamma=1.0):
    
    return np.uint8(cv2.pow(img / 255., gamma)*255.)


def augment_random_gamma(img, gammas):
    #gamma = random.uniform(*gammas)
    gamma = gammas
    return adjust_gamma(img, gamma)

def augment_random_crop(img, size):

    if not isinstance(size, tuple):
        size = (size, size)
    
    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w)/2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h)/2)
    img_pad = cv2.copyMakeBorder(img,pad_h,pad_h,pad_w,pad_w,cv2.BORDER_CONSTANT,value=[0,0,0])

    w, h = img_pad.shape[1::-1]
    if w == size[0] and h == size[1]:
        x1 = 0
        y1 = 0
    else:
        x1 = random.randint(0, w - size[0])
        y1 = random.randint(0, h - size[1])

    img_pad = img_pad[y1:y1+size[1], x1:x1+size[0], :]

    return img_pad


def augment_random_linear(img, sr=5, ssx=0.1, ssy=0.1, inter=cv2.INTER_LINEAR):

    img = img.copy()

    rot = (np.random.rand(1)[0]*2-1)*sr
    scalex = np.random.rand(1)[0]*ssx
    scaley = np.random.rand(1)[0]*ssy
    
    R = np.array([np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot)), 0, 
                  -np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0,
                  0, 0, 1
                 ]).reshape((3,3))
    
    S = np.array([1, scalex, 0, 
                  scaley, 1, 0,
                 0, 0, 1]).reshape((3,3))
    
    
    A = np.dot(R, S)

    return cv2.warpAffine(img, A.T[:2, :], img.shape[1::-1], inter, borderMode=cv2.BORDER_REFLECT)


def augment_jpeg_compress(img, qual):

    img = img.copy()
    #quality = random.randint(*quals)
    quality = qual
    enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc[1], cv2.IMREAD_COLOR)
    
    return dec

def augment_random_jpeg_compress(img, quals):

    img = img.copy()
    quality = random.randint(*quals)
    enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc[1], cv2.IMREAD_COLOR)
    
    return dec

def augment_scale(img, scale, inter=cv2.INTER_LINEAR):
    img = img.copy()
    w, h = img.shape[1::-1]
    #scale_factor = random.uniform(*scales)
    scale_factor = scale
    img = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=inter)
        
    return img

def augment_random_scale(img, scales, inter=cv2.INTER_LINEAR):
    img = img.copy()
    w, h = img.shape[1::-1]
    scale_factor = random.uniform(*scales)
    img = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=inter)
        
    return img



def magic_filter(img):
    img = img.copy()
    kernel = np.array([[-1,2,-2,2,-1], [2,-6,8,-6,2], [-2,8,-12,8,-2], [2,-6,8,-6,2], [-1,2,-2,2,-1]], np.float32)/12
    img_f = cv2.filter2D(img, -1, kernel)
    border = np.floor(np.array(kernel.shape)/2).astype(np.int)
    img_f = img_f[border[0]:-border[0],border[1]:-border[1],:]
    # normalization
    normalized_img = img_f.copy()
    cv2.normalize(img_f, normalized_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_img



def rotate_90(img):
    res = np.rot90(img, random.choice(range(2)), (0,1)).copy()
    return res

def five_crop(img, size):
    img = img.copy()
    w, h = img.shape[1::-1]
    # get central crop
    c_cr = center_crop(img, size)
    # upper-left crop
    ul_cr = img[0:size, 0:size]
    # upper-right crop
    ur_cr = img[0:size, w-size:w]
    # bottom-left crop
    bl_cr = img[h-size:h, 0:size]
    # bottom-right crop
    br_cr = img[h-size:h, w-size:w]
    return np.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))
