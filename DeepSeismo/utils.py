import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from shutil import copy2
import os
import re


# TODO: use this to encode the labels??
def one_hot(labels, n_classes=9):
    """ One_Hot """
    expansion = np.eye(n_classes)
    y = expansion[:, labels].T

    return y


def encode_labels(labels):
    """ One-hot label encoding """
    label_map = ('AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE')
    labels_encoded = np.zeros(shape=(labels.shape[0], 9), dtype=int)
    for i in range(labels.shape[0]):
        for j in range(9):
            if label_map[j] in labels[i]:
                labels_encoded[i][j] = 1
            else:
                labels_encoded[i][j] = 0
    return labels_encoded


def multiclass_f1(true, pred, return_list=False):
    """ Calculates the f1-score for each class and returns the average f1-score """

    y_pred = pred.copy()
    y_true = true.copy()

    y_pred = np.where(y_pred > 0.5, 1, 0)

    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    f1_list = []
    for m in conf_matrix:
        f1_list.append((2 * m[1, 1]) / ((2 * m[1, 1]) + m[0, 1] + m[1, 0]))

    if return_list:
        return np.mean(f1_list), f1_list

    return np.mean(f1_list)


def custom_multiclass_f1(true, pred, return_list=False):
    """ Calculates the global f1-score following the PhysioNet/CinC 2020 challenge criteria """

    # accuracies of each individual prediction
    y_pred = pred.copy()
    y_true = true.copy()

    y_pred = np.where(y_pred > 0.5, 1, 0)

    # explicitly
    from sklearn.metrics import multilabel_confusion_matrix
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    f1_list = []
    for m in conf_matrix:
        f1_list.append((5 * m[1, 1]) / ((5 * m[1, 1]) + m[0, 1] + (4*m[1, 0])))

    if return_list:
        return np.mean(f1_list), f1_list

    return np.mean(f1_list)


# draft
def split_data_train_test():
    """ Split the data files on train/test sets """

    data_dir = 'data/Training_WFDB'
    train_dir = 'data/train'
    test_dir = 'data/test'
    split = 0.2

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    subjects = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                print(file)

                # subject id
                subj_id = re.match(r"^(\w+).mat", file).group(1)
                subjects.append(subj_id)

    np.random.shuffle(subjects)

    test_size = int(len(subjects) * split)

    test_subjects = [subjects[i] for i in np.random.choice(len(subjects), test_size)]
    train_subjects = [subj for subj in subjects if subj not in test_subjects]

    for subj in test_subjects:
        print(subj)
        copy2(os.path.join(data_dir, f"{subj}.mat"), test_dir)
        copy2(os.path.join(data_dir, f"{subj}.hea"), test_dir)

    for subj in train_subjects:
        print(subj)
        copy2(os.path.join(data_dir, f"{subj}.mat"), train_dir)
        copy2(os.path.join(data_dir, f"{subj}.hea"), train_dir)


# draft
def split_train_validation_part_2(subject_predictions, subject_labels, split=0.33):
    """ Splits train/validation sets for the model_1_part_2"""

    n_timesteps = [len(v) for v in subject_predictions.values()][0]
    n_variables = 9
    n_outputs = 9

    assert len(subject_labels) == len(subject_predictions), "Labels and predictions have different shapes"

    subjects = [k for k in subject_labels.keys()]
    np.random.shuffle(subjects)
    index_split = int(len(subjects) * split)
    subjects_train = subjects[index_split:]
    subjects_validation = subjects[:index_split]

    # keep the pairwise order of subjects between labels-predictions

    # training
    X_train = np.zeros((len(subjects_train), n_timesteps, n_variables))
    y_train = np.zeros((len(subjects_train), n_outputs))
    for i, subj in enumerate(subjects_train):
        X_train[i, :, :] = subject_predictions[subj]
        y_train[i, :] = subject_labels[subj]

    # validation
    X_validation = np.zeros((len(subjects_validation), n_timesteps, n_variables))
    y_validation = np.zeros((len(subjects_validation), n_outputs))
    for i, subj in enumerate(subjects_validation):
        X_validation[i, :, :] = subject_predictions[subj]
        y_validation[i, :] = subject_labels[subj]

    return X_train, y_train, X_validation, y_validation
