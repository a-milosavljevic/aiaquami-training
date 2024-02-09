"""
data.py script is by traing.py to deal with data.
You do not have to run this script.
"""
from settings import *
import time
import random
import numpy as np
import cv2 as cv
import pandas as pd
import threading
import multiprocessing


classes = [fname for fname in os.listdir(train_folder)]
print(classes)

train_images_orig_grouped = []
train_images_grouped = []
val_images_grouped = []
test_images_grouped = []
train_max_cnt = 0
for cls in classes:
    train_cls_folder = os.path.join(train_folder, cls)
    val_cls_folder = os.path.join(val_folder, cls)
    test_cls_folder = os.path.join(test_folder, cls)
    train_cls_images = [os.path.join(train_cls_folder, fname) for fname in os.listdir(train_cls_folder)]
    val_cls_images = [os.path.join(val_cls_folder, fname) for fname in os.listdir(val_cls_folder)]
    test_cls_images = [os.path.join(test_cls_folder, fname) for fname in os.listdir(test_cls_folder)]
    if join_test_with_train:
        train_cls_images = train_cls_images + test_cls_images
        test_cls_images = []
    train_images_orig_grouped.append(train_cls_images)
    train_images_grouped.append(train_cls_images)
    val_images_grouped.append(val_cls_images)
    test_images_grouped.append(test_cls_images)
    if train_max_cnt < len(train_cls_images):
        train_max_cnt = len(train_cls_images)

for i in range(len(classes)):
    train_cls_images = train_images_grouped[i]
    if len(train_cls_images) > 0:
        while len(train_cls_images) < train_max_cnt:
            train_cls_images = train_cls_images + train_cls_images
        train_cls_images = train_cls_images[:train_max_cnt]
        train_images_grouped[i] = train_cls_images
    else:
        print(classes[i], 'has no train images!')

train_images_orig = []
train_classes_orig = []
train_images = []
train_classes = []
val_images = []
val_classes = []
test_images = []
test_classes = []
for i in range(len(classes)):
    train_images_orig = train_images_orig + train_images_orig_grouped[i]
    train_classes_orig = train_classes_orig + [i for _ in range(len(train_images_orig_grouped[i]))]
    train_images = train_images + train_images_grouped[i]
    train_classes = train_classes + [i for _ in range(len(train_images_grouped[i]))]
    val_images = val_images + val_images_grouped[i]
    val_classes = val_classes + [i for _ in range(len(val_images_grouped[i]))]
    test_images = test_images + test_images_grouped[i]
    test_classes = test_classes + [i for _ in range(len(test_images_grouped[i]))]

print(len(train_images_orig), len(train_classes_orig), len(train_images), len(train_classes),
      len(val_images), len(val_classes), len(test_images), len(test_classes))

train_orig_zip = list(zip(train_images_orig, train_classes_orig))
random.Random(0).shuffle(train_orig_zip)
train_images_orig, train_classes_orig = zip(*train_orig_zip)
#print(train_images_orig)
#print(train_classes_orig)

train_zip = list(zip(train_images, train_classes))
random.Random(0).shuffle(train_zip)
train_images, train_classes = zip(*train_zip)
#print(train_images)
#print(train_classes)

val_zip = list(zip(val_images, val_classes))
random.Random(0).shuffle(val_zip)
val_images, val_classes = zip(*val_zip)
#print(val_images)
#print(val_classes)

if not join_test_with_train:
    test_zip = list(zip(test_images, test_classes))
    random.Random(0).shuffle(test_zip)
    test_images, test_classes = zip(*test_zip)
    #print(test_images)
    #print(test_classes)


"""
# LOAD DATA

x_train = np.zeros((len(train_images), image_size, image_size, 3), dtype=np.uint8)
y_train = np.array(train_classes, dtype=np.int32)
for i in range(len(train_images)):
    img = cv.imread(train_images[i])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x_train[i] = img

x_val = np.zeros((len(val_images), image_size, image_size, 3), dtype=np.uint8)
y_val = np.array(val_classes, dtype=np.int32)
for i in range(len(val_images)):
    img = cv.imread(val_images[i])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x_val[i] = img

if not join_test_with_train:
    x_test = np.zeros((len(test_images), image_size, image_size, 3), dtype=np.uint8)
    y_test = np.array(test_classes, dtype=np.int32)
    for i in range(len(test_images)):
        img = cv.imread(test_images[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x_test[i] = img
"""


# DATA AUGMENTATION

def get_data_generator():
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                               samplewise_center=False,
                                                               featurewise_std_normalization=False,
                                                               samplewise_std_normalization=False,
                                                               zca_whitening=False,
                                                               zca_epsilon=1e-06,
                                                               rotation_range=90,
                                                               width_shift_range=0.15,
                                                               height_shift_range=0.15,
                                                               brightness_range=(0.7, 1.3),
                                                               shear_range=0.1,
                                                               zoom_range=0.2,
                                                               channel_shift_range=0.0,  # ???
                                                               fill_mode='constant',  # 'nearest',
                                                               cval=0,
                                                               horizontal_flip=True,
                                                               vertical_flip=True,
                                                               rescale=None,
                                                               preprocessing_function=None,
                                                               validation_split=0.0,
                                                               dtype=np.float32)
    return data_gen


# MULTITHREADED IMAGE LOADING
request_thread_stop = False
request_queue = []
request_queue_lock = threading.Lock()
result_queue = []
result_queue_lock = threading.Lock()


def thread_function(thread_index):
    # print("Starting thread: ", thread_index)
    data_gen = get_data_generator()
    loop = True
    while loop:
        request = None
        request_queue_lock.acquire()
        if request_thread_stop:
            loop = False
        if len(request_queue) > 0:
            request = request_queue.pop()
        request_queue_lock.release()

        if request is not None:
            idx, data_aug, fpath = request
            #print(thread_index, idx, fpath)

            image = cv.imread(fpath)
            if data_aug:
                image = data_gen.random_transform(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            result = (idx, image)
            result_queue_lock.acquire()
            result_queue.append(result)
            result_queue_lock.release()
        else:
            time.sleep(0.01)


def stop_threads():
    global request_thread_stop
    request_queue_lock.acquire()
    request_thread_stop = True
    request_queue_lock.release()


threads = []
thread_count = multiprocessing.cpu_count()
for ti in range(thread_count):
    thread = threading.Thread(target=thread_function, args=(ti,))
    threads.append(thread)
    thread.start()


# DATA GENERATOR
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images, image_classes, use_augmentation=False):
        self.images = images
        self.image_classes = image_classes
        self.use_augmentation = use_augmentation

    def __len__(self):
        return int(np.ceil(len(self.images) / batch_size))

    def __getitem__(self, idx):
        global request_queue_lock, request_queue, result_queue_lock, result_queue

        batch_start = idx * batch_size
        batch_end = min(len(self.images), (idx + 1) * batch_size)

        batch_images = self.images[batch_start:batch_end]
        batch_classes = self.image_classes[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), image_size, image_size, 3), dtype=np.float32)
        batch_y = np.array(batch_classes, dtype=np.int32)

        request_queue_lock.acquire()
        for idx in range(len(batch_images)):
            request = (idx, self.use_augmentation, batch_images[idx])
            request_queue.append(request)
        request_queue_lock.release()

        wait = True
        while wait:
            time.sleep(0.01)
            result_queue_lock.acquire()
            if len(result_queue) >= len(batch_images):
                wait = False
                for result in result_queue:
                    idx, image = result
                    batch_x[idx] = image
                result_queue = []
            result_queue_lock.release()

        return batch_x, batch_y


def test_data_augmentation(x):
    data_gen = get_data_generator()
    for i in range(len(x)):
        img = data_gen.random_transform(x[i])
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv.imwrite(os.path.join(tmp_folder, str(i) + '.jpg'), img)


########################################################################################################################
# LOCAL EXECUTION (TEST DATA AUGMENTATION)
########################################################################################################################

if __name__ == "__main__":
    img = cv.imread(train_images[0])
    n = 100
    imgs = np.zeros((n, img.shape[0], img.shape[1], 3))
    for i in range(n):
        imgs[i] = img
    test_data_augmentation(imgs)

    stop_threads()
