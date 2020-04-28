from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Resize, CenterCrop, RandomCrop
)
import numpy as np

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


from albumentations.pytorch import ToTensor

def get_train_transforms():
    augmentations = Compose([
        Resize(236,236),
        Flip(),
        OneOf([
            IAAAdditiveGaussianNoise(p=.5),
            GaussNoise(p=.4),
        ], p=0.4),
        OneOf([
            MotionBlur(p=0.6),
            Blur(blur_limit=3, p=0.2),
        ], p=0.4),
        ShiftScaleRotate(shift_limit=0.0725, scale_limit=0.2, rotate_limit=45, p=0.6),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.4),
            IAAPiecewiseAffine(p=0.2),
        ], p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.25),
        HueSaturationValue(p=0.3),
        CenterCrop(224,224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensor()
    ])

    return lambda img:augmentations(image=np.array(img))
    #return augmentations

def get_test_transforms():
    augmentations = Compose([
        Resize(236,236),
        CenterCrop(224,224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensor()
    ])
    return lambda img:augmentations(image=np.array(img))
    #return augmentations

def l(f):
    im
