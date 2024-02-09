"""
visualize_heatmaps.py script is by traing.py to generate Grad-CAM heatmaps.
You do not have to run this script.
"""
from model import *
import math


def generate_heatmaps(heatmaps_folder, images, image_classes, model, target_layer, batch_size_vis):
    batch_count = math.ceil(len(images) / batch_size_vis)
    for i in range(batch_count):
        print('Processing batch {} of {}'.format(i + 1, batch_count))

        batch_start = i * batch_size_vis
        batch_end = min(len(images), (i + 1) * batch_size_vis)

        batch_images = images[batch_start:batch_end]
        batch_classes = image_classes[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), image_size, image_size, 3), dtype=np.float32)
        batch_y_exp = np.array(batch_classes, dtype=np.int32)

        for idx in range(len(batch_images)):
            image = cv.imread(batch_images[idx])
            batch_x[idx] = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        batch_y = model.predict_on_batch(batch_x)
        batch_y_class = np.argmax(batch_y, axis=1)
        batch_y_certainty = batch_y.max(axis=1)

        # TEST
        indices = np.expand_dims(np.arange(len(batch_y_class)), axis=1)
        indices = np.concatenate((indices, np.expand_dims(batch_y_class, axis=1)), axis=1)
        output = tf.gather_nd(model.output, indices)
        grads = tf.keras.backend.gradients(output, target_layer.output)[0]
        pooled_grads = tf.keras.backend.mean(grads, axis=(1, 2))
        iterate = tf.keras.backend.function([model.input], [pooled_grads, target_layer.output])
        pooled_grads_value, target_layer_output_value = iterate([batch_x])

        pooled_grads_value_ex = np.expand_dims(pooled_grads_value, axis=1)
        pooled_grads_value_ex = np.expand_dims(pooled_grads_value_ex, axis=2)
        pooled_grads_value_ex = np.repeat(pooled_grads_value_ex, target_layer_output_value.shape[1], axis=1)
        pooled_grads_value_ex = np.repeat(pooled_grads_value_ex, target_layer_output_value.shape[2], axis=2)
        target_layer_output_value = np.multiply(target_layer_output_value, pooled_grads_value_ex)

        heatmaps = np.mean(target_layer_output_value, axis=-1)
        heatmaps = np.maximum(heatmaps, 0)
        heatmaps_max = np.max(heatmaps, axis=(1, 2))
        heatmaps_max = np.expand_dims(heatmaps_max, axis=1)
        heatmaps_max = np.expand_dims(heatmaps_max, axis=2)
        heatmaps_max = np.repeat(heatmaps_max, heatmaps.shape[1], axis=1)
        heatmaps_max = np.repeat(heatmaps_max, heatmaps.shape[2], axis=2)
        heatmaps = np.divide(heatmaps, heatmaps_max)

        for bi in range(len(batch_y_class)):
            print('{0:5d}. {1}'.format(bi + 1, images[batch_start + bi]))

            img = batch_x[bi].copy()
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            heatmap = heatmaps[bi]

            heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

            if False:
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                img = np.expand_dims(img, axis=2)
                img = np.repeat(img, 3, axis=2)
            superimposed_img = heatmap * 0.4 + img
            fname = os.path.basename(images[batch_start + bi])
            prefix = '[{} as {} {}%] '.format(classes[batch_y_exp[bi]], classes[batch_y_class[bi]],
                                              round(100 * batch_y_certainty[bi]))
            cv.imwrite(os.path.join(heatmaps_folder, prefix + fname[:-4] + '.jpg'), superimposed_img)


########################################################################################################################
# LOCAL EXECUTION
########################################################################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    # Prepare output folders
    heatmaps_test_folder = 'heatmaps_test'
    heatmaps_test_folder = os.path.join(tmp_folder, heatmaps_test_folder)
    if not os.path.exists(heatmaps_test_folder):
        os.mkdir(heatmaps_test_folder)
    if not heatmaps_for_test_images_only:
        heatmaps_train_folder = 'heatmaps_train'
        heatmaps_val_folder = 'heatmaps_val'
        heatmaps_train_folder = os.path.join(tmp_folder, heatmaps_train_folder)
        heatmaps_val_folder = os.path.join(tmp_folder, heatmaps_val_folder)
        if not os.path.exists(heatmaps_train_folder):
            os.mkdir(heatmaps_train_folder)
        if not os.path.exists(heatmaps_val_folder):
            os.mkdir(heatmaps_val_folder)

    # LOAD MODEL
    model_path = os.path.join(tmp_folder, 'model.h5')
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    model.summary()

    # EVALUATE MODEL
    if False:
        eval_gen_train = DataGenerator(images=train_images_orig, image_classes=train_classes_orig,
                                       use_augmentation=False)
        res = model.evaluate(eval_gen_train)
        print('EVAL TRAIN:', res)

        eval_gen_valid = DataGenerator(images=val_images, image_classes=val_classes, use_augmentation=False)
        res = model.evaluate(eval_gen_valid)
        print('EVAL VAL:', res)

        if not join_test_with_train:
            eval_gen_test = DataGenerator(images=test_images, image_classes=test_classes, use_augmentation=False)
            res = model.evaluate(eval_gen_test)
            print('EVAL TEST:', res)

    # GET TARGET LAYER
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
