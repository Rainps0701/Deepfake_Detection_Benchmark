import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np

DetectionTests = {
    'Celeb-DF': {
        'dataroot': './dataset/test/Celeb-DF',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    'DFD': {
        'dataroot': './dataset/test/DFD',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    'DFDC': {
        'dataroot': './dataset/test/DFDC',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    'DFDCP': {
        'dataroot': './dataset/test/DFDCP',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    'FF++': {
        'dataroot': './dataset/test/FF++',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    'UADFA': {
        'dataroot': './dataset/test/UADFA',
        'no_resize': False,  # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    }
}

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# Load model
model = freqnet(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()

# Iterate over test datasets
for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    # Update dataroot in options
    opt.dataroot = dataroot

    # Validate and get metrics
    acc, ap, auc, f1, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    # Print metrics for the current test set
    print(f"Results for {testSet}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy of Real Images (RACC): {r_acc:.4f}")
    print(f"Accuracy of Fake Images (FACC): {f_acc:.4f}")
    print('*' * 50)
