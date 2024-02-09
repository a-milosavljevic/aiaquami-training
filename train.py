"""
train.py script is used to train the model based on specified settings.
In addition to training it also creates confusion matrices and Grad-CAM heatmaps.
"""
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from confusion_matrix import plot_confusion_matrix
from visualize_heatmaps import generate_heatmaps
import json
import math


########################################################################################################################
# MODEL
########################################################################################################################

model = create_model(False)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()


########################################################################################################################
# CALLBACKS
########################################################################################################################

model_path = os.path.join(curr_folder, 'model.h5')
metadata_path = os.path.join(curr_folder, 'model.json')
log_path = os.path.join(curr_folder, 'log.csv')
log_warmup_path = os.path.join(curr_folder, 'log_warmup.csv')
eval_path = os.path.join(curr_folder, 'eval.txt')


def cosine_annealing_schedule(t, lr, period=lr_period, scale=lr_scale, decay=lr_decay):
    step = t // period
    lr = init_lr * (decay**step)
    arg = np.pi * (t % period) / period
    k = (np.cos(arg) + 1) / 2
    k = k * (1 - scale) + scale
    ret = float(lr * k)
    print('cosine_annealing_schedule(t={}, lr={}, period={}, scale={}) -> {}'.format(t, lr, period, scale, ret))
    return ret


csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)
csv_logger_warmup = tf.keras.callbacks.CSVLogger(log_warmup_path, separator=',', append=False)
schedule_lr = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_schedule, verbose=0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=reduce_lr_patience,
                                                 monitor=val_monitor[0], mode=val_monitor[1])
early_stopping = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1,
                                                  restore_best_weights=True,
                                                  monitor=val_monitor[0], mode=val_monitor[1])

lr_callback = schedule_lr if cosine_annealing else reduce_lr


########################################################################################################################
# TRAINING
########################################################################################################################

data_gen_train = DataGenerator(images=train_images, image_classes=train_classes, use_augmentation=data_augmentation)
data_gen_valid = DataGenerator(images=val_images, image_classes=val_classes, use_augmentation=data_augmentation)

# Warm up
hist_warmup = model.fit(data_gen_train,
                        epochs=epochs_warmup,
                        validation_data=data_gen_valid,
                        shuffle=True,
                        callbacks=[csv_logger_warmup, lr_callback],
                        verbose=1)

# Full train
for layer in model.layers:
    layer.trainable = True
hist = model.fit(data_gen_train,
                 epochs=epochs,
                 validation_data=data_gen_valid,
                 shuffle=True,
                 callbacks=[csv_logger, lr_callback, early_stopping],
                 verbose=1)

# Save model and model metadata
model.save(model_path, include_optimizer=False)
metadata = dict()
metadata['classes'] = classes
metadata['image_size'] = image_size
metadata['stretch'] = stretch
metadata['encoder'] = architecture
metadata['last_conv_layer'] = encoder_last_conv_layer
metadata['dropout_rate'] = dropout_rate
metadata['hidden_neurons'] = hidden_neurons
metadata['optimizer'] = optimizer_name
metadata['init_learning_rate'] = init_lr
metadata['batch_size'] = batch_size
metadata['data_augmentation'] = data_augmentation
metadata['cosine_annealing'] = cosine_annealing
metadata['monitor_loss'] = monitor_loss
json_metadata = json.dumps(metadata, indent=2)
with open(metadata_path, 'w') as f:
    f.write(json_metadata)

# Save training graphs
plt.clf()
plt.plot(hist_warmup.history['loss'])
plt.plot(hist_warmup.history['val_loss'])
plt.savefig(os.path.join(curr_folder, 'loss_warmup.png'), dpi=300)

plt.clf()
plt.plot(hist_warmup.history['sparse_categorical_accuracy'])
plt.plot(hist_warmup.history['val_sparse_categorical_accuracy'])
plt.savefig(os.path.join(curr_folder, 'accuracy_warmup.png'), dpi=300)

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.savefig(os.path.join(curr_folder, 'loss.png'), dpi=300)

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.savefig(os.path.join(curr_folder, 'accuracy.png'), dpi=300)

# Evaluate model
train_acc = 0
val_acc = 0
test_acc = 0
with open(eval_path, 'w') as f:
    eval_gen_train = DataGenerator(images=train_images_orig, image_classes=train_classes_orig, use_augmentation=False)
    res = model.evaluate(eval_gen_train)
    print('EVAL TRAIN:', res)
    f.write('Train loss: {}\r\n'.format(res[0]))
    f.write('Train acc: {}\r\n'.format(res[1]))
    train_acc = res[1]

    eval_gen_valid = DataGenerator(images=val_images, image_classes=val_classes, use_augmentation=False)
    res = model.evaluate(eval_gen_valid)
    print('EVAL VAL:', res)
    f.write('Val loss: {}\r\n'.format(res[0]))
    f.write('Val acc: {}\r\n'.format(res[1]))
    val_acc = res[1]

    if not join_test_with_train:
        eval_gen_test = DataGenerator(images=test_images, image_classes=test_classes, use_augmentation=False)
        res = model.evaluate(eval_gen_test)
        print('EVAL TEST:', res)
        f.write('Test loss: {}\r\n'.format(res[0]))
        f.write('Test acc: {}\r\n'.format(res[1]))
        test_acc = res[1]

# Write entry in experiments.csv
with open(experiments_file, 'a') as f:
    f.write('{},{}{},{},{},{},{},{},{},{},{},{},{},{},{}\r\n'.format(experiment_no,
                                                                     'st' if stretch else '',
                                                                     image_size,
                                                                     architecture,
                                                                     dropout_rate,
                                                                     hidden_neurons,
                                                                     data_augmentation,
                                                                     cosine_annealing,
                                                                     init_lr,
                                                                     optimizer_name,
                                                                     monitor_loss,
                                                                     batch_size,
                                                                     train_acc,
                                                                     val_acc,
                                                                     test_acc))

# Rename current folder
curr_folder_new = curr_folder + ' ({})'.format(test_acc)
os.rename(curr_folder, curr_folder_new)
curr_folder = curr_folder_new

# Generate confusion matrices
if True:
    if join_test_with_train:
        cm_images = val_images
        cm_classes = val_classes
    else:
        cm_images = test_images
        cm_classes = test_classes

    cm_gen = DataGenerator(images=cm_images, image_classes=cm_classes, use_augmentation=False)
    y_cm_raw = model.predict(cm_gen)
    y_cm_out = np.argmax(y_cm_raw, axis=1)
    y_cm = np.array(cm_classes, dtype=np.int32)
    misclassified = np.sum(y_cm != y_cm_out)
    accuracy = round(100 * (len(y_cm) - misclassified) / len(y_cm), 2)

    cm = confusion_matrix(y_cm, y_cm_out)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices
    np.set_printoptions(precision=1)

    cm = confusion_matrix(y_cm, y_cm_out)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig_size = round(len(classes) * 0.5)
    fig_size = (5 + fig_size, fig_size)

    plot_confusion_matrix(y_cm, y_cm_out, figsize=fig_size,
                          title='Misclassified {} out of {} specimens'.format(misclassified, len(y_cm)))
    plt.savefig(os.path.join(curr_folder, 'cm.png'))

    plot_confusion_matrix(y_cm, y_cm_out, normalize=True, figsize=fig_size,
                          title='Classification accuracy {}%'.format(accuracy))
    plt.savefig(os.path.join(curr_folder, 'cm_norm.png'))

# Generate heatmaps
if True:
    model = None
    tf.compat.v1.disable_eager_execution()

    heatmaps_test_folder = 'heatmaps_test'
    heatmaps_test_folder = os.path.join(curr_folder, heatmaps_test_folder)
    if not os.path.exists(heatmaps_test_folder):
        os.mkdir(heatmaps_test_folder)
    if not heatmaps_for_test_images_only:
        heatmaps_train_folder = 'heatmaps_train'
        heatmaps_val_folder = 'heatmaps_val'
        heatmaps_train_folder = os.path.join(curr_folder, heatmaps_train_folder)
        heatmaps_val_folder = os.path.join(curr_folder, heatmaps_val_folder)
        if not os.path.exists(heatmaps_train_folder):
            os.mkdir(heatmaps_train_folder)
        if not os.path.exists(heatmaps_val_folder):
            os.mkdir(heatmaps_val_folder)

    # Load model since eager execution disabled
    model = tf.keras.models.load_model(os.path.join(curr_folder, 'model.h5'))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    # Get target layer
    target_layer = model.get_layer(encoder_last_conv_layer)

    batch_size_vis = 4 * batch_size

    print("PROCESSING TEST IMAGES")
    generate_heatmaps(heatmaps_test_folder, test_images, test_classes,
                      model, target_layer, batch_size_vis)
    if not heatmaps_for_test_images_only:
        print("PROCESSING VALIDATION IMAGES")
        generate_heatmaps(heatmaps_val_folder, val_images, val_classes,
                          model, target_layer, batch_size_vis)
        print("PROCESSING TRAIN IMAGES")
        generate_heatmaps(heatmaps_train_folder, train_images_orig, train_classes_orig,
                          model, target_layer, batch_size_vis)

stop_threads()
