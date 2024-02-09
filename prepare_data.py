"""
prepare_data.py script is used to process original images and create training, validation, and test subsets.
The output depends from image_size and stretch parameters, so the script should be executed initially and
after changing those two parameters.
"""
from settings import *
import numpy as np
import cv2 as cv
import random


def square_resize_image(img_input):
    img_out = img_input
    if not stretch:
        w, h = img_out.shape[1], img_out.shape[0]
        top, bottom, left, right = 0, 0, 0, 0
        if w >= h:
            top = (w - h) // 2
            bottom = (w - h) - top
        else:
            left = (h - w) // 2
            right = (h - w) - left
        #mean_color = tuple(np.average(img_input, axis=(0, 1)))
        img_out = cv.copyMakeBorder(img_out, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0)) #mean_color
    img_out = cv.resize(img_out, (image_size, image_size), interpolation=cv.INTER_AREA)
    return img_out


classes = [fname for fname in os.listdir(original_data_folder) if len(fname) == 3]
print(len(classes), classes)

for cls in classes:
    cls_folder = os.path.join(original_data_folder, cls)
    files = [fname for fname in os.listdir(cls_folder)]
    train_images = []
    val_images = []
    test_images = []
    for fname in files:
        if len(fname) < 14 or len(fname) > 15:
            print('BAD FILENAME:', fname)
        if fname[-4:].lower() == '.jpg' or fname[-4:].lower() == '.tif':
            #print(fname, fname[6:9])
            unit = int(fname[6:9])
            fold = unit % len(subset_distribution)
            if subset_distribution[fold].upper() == 'T':
                train_images.append(fname)
            elif subset_distribution[fold].upper() == 'V':
                val_images.append(fname)
            else:
                test_images.append(fname)

    # patch for small number of samples per classes
    while len(train_images) < min_train_images and len(val_images) > 0:
        unit = val_images[0][6:9]
        new_val_images = []
        for fname in val_images:
            if fname[6:9] == unit:
                train_images.append(fname)
            else:
                new_val_images.append(fname)
        val_images = new_val_images

    loop = True
    while loop and len(train_images) < min_train_images and len(train_images) < 5 * len(test_images) and len(test_images) > 1:
        unit = test_images[0][6:9]
        new_train_images = []
        new_test_images = []
        for fname in test_images:
            if fname[6:9] == unit:
                new_train_images.append(fname)
            else:
                new_test_images.append(fname)
        if len(new_test_images) > 0 or len(train_images) == 0:
            train_images = train_images + new_train_images
            test_images = new_test_images
        else:
            loop = False

    print(len(train_images), len(val_images), len(test_images), cls)

    if False: #True:
        # EXTRACT TEST IMAGES FROM ORIGINAL DATASET
        orig_test_folder = os.path.join(data_folder, 'test')
        orig_test_cls_folder = os.path.join(orig_test_folder, cls)
        if not os.path.exists(orig_test_folder):
            os.makedirs(orig_test_folder)
        if not os.path.exists(orig_test_cls_folder):
            os.makedirs(orig_test_cls_folder)

        for fname in test_images:
            fpath = os.path.join(cls_folder, fname)
            shutil.copyfile(fpath, os.path.join(orig_test_cls_folder, fname))
    else:
        # NORMAL PREPROCESSING FOR TRAINING PURPOSES
        train_cls_folder = os.path.join(train_folder, cls)
        if not os.path.exists(train_cls_folder):
            os.makedirs(train_cls_folder)
        val_cls_folder = os.path.join(val_folder, cls)
        if not os.path.exists(val_cls_folder):
            os.makedirs(val_cls_folder)
        test_cls_folder = os.path.join(test_folder, cls)
        if not os.path.exists(test_cls_folder):
            os.makedirs(test_cls_folder)

        for fname in train_images:
            fpath = os.path.join(cls_folder, fname)
            img = cv.imread(fpath)
            img = square_resize_image(img)
            cv.imwrite(os.path.join(train_cls_folder, fname[:-4] + '.jpg'), img)

        for fname in val_images:
            fpath = os.path.join(cls_folder, fname)
            img = cv.imread(fpath)
            img = square_resize_image(img)
            cv.imwrite(os.path.join(val_cls_folder, fname[:-4] + '.jpg'), img)

        for fname in test_images:
            fpath = os.path.join(cls_folder, fname)
            img = cv.imread(fpath)
            img = square_resize_image(img)
            cv.imwrite(os.path.join(test_cls_folder, fname[:-4] + '.jpg'), img)
