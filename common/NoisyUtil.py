import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal


def gen_subclass_noise_20(target):
    # One-fifths of the sub-classes are flipped into noisy classes.
    if target in [4, 30, 55, 72, 89]:
        return 0
    elif target in [1, 32, 67, 73, 95]:
        return 1
    elif target in [54, 62, 70, 82, 91]:
        return 2
    elif target in [9, 10, 16, 28, 92]:
        return 3
    elif target in [0, 51, 53, 57, 61]:
        return 4
    elif target in [22, 39, 40, 86, 83]:
        return 5
    elif target in [5, 20, 25, 84, 87]:
        return 6
    elif target in [6, 7, 14, 18, 94]:
        return 7
    elif target in [3, 42, 43, 88, 24]:
        return 8
    elif target in [12, 17, 37, 68, 97]:
        return 9
    elif target in [23, 33, 49, 60, 76]:
        return 10
    elif target in [15, 19, 21, 31, 71]:
        return 11
    elif target in [34, 63, 64, 66, 38]:
        return 12
    elif target in [26, 45, 77, 79, 75]:
        return 13
    elif target in [2, 11, 35, 46, 99]:
        return 14
    elif target in [27, 29, 44, 78, 98]:
        return 15
    elif target in [36, 50, 65, 74, 93]:
        return 16
    elif target in [47, 52, 56, 59, 80]:
        return 17
    elif target in [8, 13, 48, 58, 96]:
        return 18
    elif target in [41, 69, 81, 85, 90]:
        return 19
    return None


def gen_subclass_noise_40(target):
    # Two-fifths of the sub-classes are flipped into noisy classes.
    # Given that CIFAR20-SDN already presents significant challenges, we do not utilize it in our experiments, reserving more complex scenarios for future research.
    if target in [4, 30, 55, 85, 89]:
        return 0
    elif target in [1, 32, 67, 72, 95]:
        return 1
    elif target in [54, 62, 70, 73, 91]:
        return 2
    elif target in [9, 10, 16, 82, 92]:
        return 3
    elif target in [0, 51, 53, 28, 61]:
        return 4
    elif target in [22, 39, 40, 57, 83]:
        return 5
    elif target in [5, 20, 25, 86, 87]:
        return 6
    elif target in [6, 7, 14, 84, 94]:
        return 7
    elif target in [3, 42, 43, 18, 24]:
        return 8
    elif target in [12, 17, 37, 88, 97]:
        return 9
    elif target in [23, 33, 49, 68, 76]:
        return 10
    elif target in [15, 19, 21, 60, 71]:
        return 11
    elif target in [34, 63, 64, 31, 38]:
        return 12
    elif target in [26, 45, 77, 66, 75]:
        return 13
    elif target in [2, 11, 35, 79, 99]:
        return 14
    elif target in [27, 29, 44, 46, 98]:
        return 15
    elif target in [36, 50, 65, 78, 93]:
        return 16
    elif target in [47, 52, 56, 74, 80]:
        return 17
    elif target in [8, 13, 48, 59, 96]:
        return 18
    elif target in [41, 69, 81, 58, 90]:
        return 19
    return None


def gen_subclean(target):
    if target in [4, 30, 55, 72, 95]:
        return 0
    elif target in [1, 32, 67, 73, 91]:
        return 1
    elif target in [54, 62, 70, 82, 92]:
        return 2
    elif target in [9, 10, 16, 28, 61]:
        return 3
    elif target in [0, 51, 53, 57, 83]:
        return 4
    elif target in [22, 39, 40, 86, 87]:
        return 5
    elif target in [5, 20, 25, 84, 94]:
        return 6
    elif target in [6, 7, 14, 18, 24]:
        return 7
    elif target in [3, 42, 43, 88, 97]:
        return 8
    elif target in [12, 17, 37, 68, 76]:
        return 9
    elif target in [23, 33, 49, 60, 71]:
        return 10
    elif target in [15, 19, 21, 31, 38]:
        return 11
    elif target in [34, 63, 64, 66, 75]:
        return 12
    elif target in [26, 45, 77, 79, 99]:
        return 13
    elif target in [2, 11, 35, 46, 98]:
        return 14
    elif target in [27, 29, 44, 78, 93]:
        return 15
    elif target in [36, 50, 65, 74, 80]:
        return 16
    elif target in [47, 52, 56, 59, 96]:
        return 17
    elif target in [8, 13, 48, 58, 90]:
        return 18
    elif target in [41, 69, 81, 85, 89]:
        return 19
    return None


def gen_subclass_noise(targets, noise_rate, noise_type=1):
    if noise_type == 1:
        # For CIFAR20-SDN, only one-fifth of the classes are flipped.
        # The input noise rate applies to each class, so it's multiplied by five.
        flips = np.random.binomial(1, noise_rate * 5, len(targets))
        # print("real noise rate", np.sum(flips) / len(targets) / 5)
    else:
        # two-fifth of the classes are flipped.
        flips = np.random.binomial(1, noise_rate * 5 / 2, len(targets))
        # print("real noise rate", np.sum(flips) / len(targets) / 5 * 2)

    new_targets = []
    for i in range(len(targets)):
        if flips[i] == 1 and noise_type == 1:
            new_targets.append(gen_subclass_noise_20(targets[i]))
        elif flips[i] == 1 and noise_type == 2:
            new_targets.append(gen_subclass_noise_40(targets[i]))
        else:
            new_targets.append(gen_subclean(targets[i]))

    return np.array(new_targets)


def dataset_split(train_images, train_labels, noise_rate=0.5, noise_type='symmetric', split_per=0.9, random_seed=1, num_classes=10, include_noise=False):

    if include_noise:
        noise_rate = noise_rate * (1 - 1 / num_classes)
        print("include_noise True, new real nosie rate:", noise_rate)

    clean_train_labels = train_labels[:, np.newaxis]
    if noise_type == 'pairflip':
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels, noise=noise_rate,
                                                                            random_state=random_seed, nb_classes=num_classes)

    elif noise_type == 'subclass':
        print("generate cifar20 with class flip label noise")
        noisy_labels = gen_subclass_noise(train_labels, noise_rate)
        clean_train_labels = gen_subclass_noise(train_labels, 0)
    else:
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels, noise=noise_rate,
                                                                                        random_state=random_seed, nb_classes=num_classes)

    clean_train_labels = clean_train_labels.squeeze()
    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]
    train_clean_labels, val_clean_labels = clean_train_labels[train_set_index], clean_train_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels, train_clean_labels, val_clean_labels


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise, P


# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # i is np.array, such as [1]
        if not isinstance(i, np.ndarray):
            i = [i]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


class Train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.length = len(self.targets)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Semi_Labeled_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.length = len(self.targets)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return out1, out2, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Semi_Unlabeled_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = np.array(data)
        self.length = self.data.shape[0]

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            out1 = self.transform(img)
            out2 = self.transform(img)

        return out1, out2

    def __len__(self):
        return self.length

    def getData(self):
        return self.data


def getNoisyData(seed, dataset, data_root, data_percent, noise_type, noise_rate, include_noise=False):
    """
    return train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels
    """
    if dataset == "CIFAR10" or dataset == "cifar10":
        num_classes = 10
        train_set = CIFAR10(root=data_root, train=True, download=False)
    elif dataset == "CIFAR100" or dataset == "cifar100" or dataset == "CIFAR20" or dataset == "cifar20":
        num_classes = 100
        train_set = CIFAR100(root=data_root, train=True, download=False)

    return dataset_split(train_set.data, np.array(train_set.targets), noise_rate, noise_type, data_percent, seed, num_classes, include_noise)
