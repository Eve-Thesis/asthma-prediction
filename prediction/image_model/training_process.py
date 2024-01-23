import logging.config
import os

import numpy as np
import tensorflow as tf
import datetime

from sklearn.metrics import confusion_matrix, roc_auc_score

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('training')

# Define metrics
train_loss = tf.keras.metrics.CategoricalCrossentropy('train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
# train_auc = tf.keras.metrics.AUC('train_AUC')
# train_precision = tf.keras.metrics.Precision('train_precision')
# train_recall = tf.keras.metrics.Precision('train_recall')

test_loss = tf.keras.metrics.CategoricalCrossentropy('test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
# test_auc = tf.keras.metrics.AUC('test_AUC')
# test_precision = tf.keras.metrics.Precision('test_precision')
# test_recall = tf.keras.metrics.Precision('test_recall')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def break_training(model, parameters=None):
    # Break training if necessary
    if os.path.exists("stop.flag"):
        print("\nIteration interrupted on request. Model:")
        print(model.summary())
        if parameters:
            print(parameters)
        while True:
            response = input("Do you want to break training? (y/n): ")
            if response == "y":
                return True
            elif response == "n":
                os.remove("stop.flag")
                return False
    return False


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    auc = roc_auc_score(y_true, y_pred)
    return tpr, tnr, f1, auc


def train_step(model, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        # predictions_avg = np.mean(predictions, axis=0)
        y_train_one = tf.convert_to_tensor(y_train[0], np.int32)
        y_train_one = tf.reshape(y_train_one, (1, 2))
        loss = model.compiled_loss(y_train_one, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    # print(model.optimizer)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(y_train_one, predictions)
    train_accuracy(y_train_one, predictions)
    predictions = tf.reshape(predictions, (2))
    return loss, predictions


def test_step(model, x_test, y_test):
    predictions = model(x_test)
    y_test_one = tf.convert_to_tensor(y_test[0], np.int32)
    y_test_one = tf.reshape(y_test_one, (1, 2))
    loss = model.compiled_loss(y_test_one, predictions)

    test_loss(y_test_one, predictions)
    test_accuracy(y_test_one, predictions)
    predictions = tf.reshape(predictions, (2))
    return loss, predictions


def evaluate_model(model, data, training=False, metrics_file=None, epoch=None):
    all_predicted = []
    all_labels = []
    all_loss = []
    # all_accuracy = []
    for sample in data:
        # sample - contains multiple images of one spectrogram and one one-hot label
        if np.shape(sample[0]) == (0, 3):  # If there is no images in one sound sample :c
            continue
        labels = np.array(sample[1])
        sample_data = sample[0]
        repeated_labels = np.repeat([labels], len(sample[0]), axis=0)

        if training:
            loss, predictions = train_step(model, sample_data, repeated_labels)
        else:
            loss, predictions = test_step(model, sample_data, repeated_labels)

        all_predicted.append(predictions)
        all_labels.append(labels)
        all_loss.append(loss)
        # all_accuracy.append(accur)

    tpr, tnr, f1, auc = calculate_metrics(np.argmax(all_labels, axis=1), np.argmax(all_predicted, axis=1))
    sample_loss = sum(all_loss) / len(all_loss)
    # sample_accuracy = sum(all_accuracy) / len(all_accuracy)

    # print(f"{model.optimizer._decayed_lr('float32').numpy():.11f}")
    # print(f"{model.optimizer.lr(model.optimizer.iterations):.11f}")
    if training:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('TPR', tpr, step=epoch)
            tf.summary.scalar('TNR', tnr, step=epoch)
            tf.summary.scalar('F1-score', f1, step=epoch)
            tf.summary.scalar('AUC', auc, step=epoch)
            tf.summary.scalar('base_LR', model.optimizer.optimizer_specs[0]['optimizer'].lr(
                model.optimizer.optimizer_specs[0]['optimizer'].iterations), step=epoch)
            tf.summary.scalar('top_LR', model.optimizer.optimizer_specs[1]['optimizer'].lr(
                model.optimizer.optimizer_specs[1]['optimizer'].iterations), step=epoch)
    else:
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            tf.summary.scalar('TPR', tpr, step=epoch)
            tf.summary.scalar('TNR', tnr, step=epoch)
            tf.summary.scalar('F1-score', f1, step=epoch)
            tf.summary.scalar('AUC', auc, step=epoch)

    metrics_info = (f"Accuracy: {train_accuracy.result()}, Loss: {sample_loss:.2f}, "
                    f"TPR: {tpr:.2f}, TNR: {tnr:.2f}, f1: {f1:.2f}, AUC: {auc:.2f}")
    logger.info(metrics_info)
    if metrics_file:
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch: {epoch}, "
                    f"{metrics_info}\n")

    return sample_loss, tpr, tnr, f1, auc


def train(model, train_data, val_data, epochs=100, start_epoch=0, patience=15, metrics_file=None, early_stopping=None,
          save_checkpoint_file=None, load_checkpoint_file=None, batch_size=1):
    best_val_tpr_tnr = 0.0  # 0.5 * (tpr + tnr)
    curr_step = 0
    rng = np.random.default_rng()
    for epoch in range(epochs - start_epoch):
        # Break training if necessary
        if break_training(model, save_checkpoint_file):
            return
        # Shuffle train_data in each epoch
        rng.shuffle(train_data, axis=0)
        logger.info(f"\nEpoch: {epoch + start_epoch}")
        # Training.
        logger.info("Training")
        evaluate_model(model, train_data, training=True, metrics_file=metrics_file, epoch=epoch + start_epoch)
        # Validation.
        logger.info("Validation")
        _, tpr, tnr, _, _ = evaluate_model(model, val_data, training=False,
                                           metrics_file=metrics_file, epoch=epoch + start_epoch)

        # Early stopping.
        # if early_stopping:
        #     early_stopping.on_epoch_end(epoch, logs={'val_tpr_tnr': 0.5 * (tpr + tnr)})
        #     if early_stopping.stopped_epoch > 0:
        #         logger.info("Early stopping. Restoring best weights.")
        #         break

        # Early stopping and save model weights.
        if early_stopping:
            curr_val_tpr_tnr = 0.5 * (tpr + tnr)
            if best_val_tpr_tnr <= curr_val_tpr_tnr:
                if tpr > 0.5 and tnr > 0.5:
                    curr_step = 0
                    if best_val_tpr_tnr < curr_val_tpr_tnr:
                        # Save the model weights to disk:
                        logger.info("Validation TPR/TNR improved. Saving best weights.")
                        model.save_weights(save_checkpoint_file)
                        best_val_tpr_tnr = curr_val_tpr_tnr
                else:
                    curr_step += 1
                    if best_val_tpr_tnr < curr_val_tpr_tnr:
                        curr_step = 0
                        best_val_tpr_tnr = curr_val_tpr_tnr
            else:
                curr_step += 1

            if curr_step >= patience:
                print("Early Stop! (Train)")
                break

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # Save best results to file.
    logger.info(f"Best val: {best_val_tpr_tnr}\n")
    if metrics_file:
        with open(metrics_file, 'a') as f:
            f.write(f"Best val: {best_val_tpr_tnr}\n")

    return model
