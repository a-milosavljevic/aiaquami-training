"""
settings.py script contains different parameters that specify model and training procedure.
experiment_no is used for training different experiments and it should be incremented before training.
"""
import os
import tensorflow as tf
import shutil


########################################################################################################################
# TRAINING SETTINGS
########################################################################################################################

experiment_no = 1

batch_size = 16

image_size = 512
stretch = False

subset_distribution = ['T', 'S', 'V', 'T', 'S', 'T', 'T', 'V', 'S', 'T']  # train (T) 50%, val (V) 20%, test (S) 30%

min_train_images = 30

#optimizer_name = 'SGD'
#optimizer_name = 'RMSprop'
optimizer_name = 'Adam'

if optimizer_name == 'SGD':
    init_lr = 1e-2
    optimizer = tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9)
elif optimizer_name == 'RMSprop':
    init_lr = 1e-3
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=init_lr)
elif optimizer_name == 'Adam':
    init_lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

data_augmentation = True
cosine_annealing = False

monitor_loss = False
if monitor_loss:
    val_monitor = ('val_loss', 'min')
else:
    val_monitor = ('val_sparse_categorical_accuracy', 'max')

lr_scale = 0.01
lr_period = 10
lr_decay = 0.7
if cosine_annealing:
    epochs = 100 * lr_period
    epochs_warmup = lr_period
    reduce_lr_patience = lr_period
    early_stopping_patience = 3 * lr_period
else:
    epochs = 1000
    epochs_warmup = 10
    reduce_lr_patience = 10
    early_stopping_patience = 30

join_test_with_train = False

heatmaps_for_test_images_only = True #False


########################################################################################################################
# MODEL SETTINGS
########################################################################################################################

dropout_rate = 0    # 0.5
hidden_neurons = 0  # 1024

#architecture = 'ResNet50'
#architecture = 'ResNet50V2'
#architecture = 'EfficientNetB0'
#architecture = 'EfficientNetB1'
architecture = 'EfficientNetB2'
#architecture = 'EfficientNetB3'
#architecture = 'EfficientNetB4'
#architecture = 'EfficientNetB5'
#architecture = 'EfficientNetB6'


########################################################################################################################
# FOLDER SETTINGS
########################################################################################################################

#root_folder = os.path.join('D:\\', 'Datasets', 'AIAQUAMI')
#root_folder = os.path.join('D:\\', 'Datasets', 'AIAQUAMI_Chiro')
root_folder = os.path.join('D:\\', 'Datasets', 'AIAQUAMI_EPT')

data_folder = os.path.join(root_folder, 'data')
original_data_folder = os.path.join(data_folder, 'data_original')
tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

train_folder = os.path.join(data_folder, 'train_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

val_folder = os.path.join(data_folder, 'val_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

test_folder = os.path.join(data_folder, 'test_{}{}'.format(image_size, '_st' if stretch else ''))
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

curr_folder_name = '{}_{}{}px_{}_do{}_hn{}_da{}_ca{}_lr{}_{}_{}_bs{}'.format('{:03d}'.format(experiment_no),
                                                                             'st' if stretch else '',
                                                                             image_size,
                                                                             architecture,
                                                                             dropout_rate,
                                                                             hidden_neurons,
                                                                             'Y' if data_augmentation else 'N',
                                                                             'Y' if cosine_annealing else 'N',
                                                                             init_lr,
                                                                             optimizer_name,
                                                                             'loss' if monitor_loss else 'acc',
                                                                             batch_size)
curr_folder = os.path.join(tmp_folder, curr_folder_name)
if not os.path.exists(curr_folder):
    os.mkdir(curr_folder)

src_settings_file = os.path.join(os.getcwd(), 'settings.py')
dst_settings_file = os.path.join(curr_folder, 'settings.py')
if os.path.exists(dst_settings_file):
    os.remove(dst_settings_file)
shutil.copyfile(src_settings_file, dst_settings_file)

experiments_file = os.path.join(tmp_folder, 'experiments.csv')
if not os.path.exists(experiments_file):
    with open(experiments_file, 'w') as f:
        f.write('ExperimentNo,ImageSize,Architecture,Dropout,HiddenNeurons,DataAugmentation,CosineAnnealing,'
                'InitLearningRate,Optimizer,MonitorLoss,BatchSize,TrainAcc,ValAcc,TestAcc\r\n')
