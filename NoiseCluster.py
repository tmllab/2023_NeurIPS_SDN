import os
import os.path
import argparse
import random
import numpy as np
from sklearn.cluster import DBSCAN
from MulticoreTSNE import MulticoreTSNE as TSNE

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR100

from common.NoisyUtil import getNoisyData, gen_subclass_noise, Train_Dataset
from common.tools import getTime, evaluate, train, predict_repre
from common.ResNet import PreActResNet18, ResNet18, ResNet34
warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay for training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--arch', default='resnet34', type=str)
parser.add_argument('--eps', default=0.02, type=float)
parser.add_argument('--min_samples', default=100, type=int)
parser.add_argument('--T1', default=80, type=int, help='stopping epoch for later stopping')
parser.add_argument('--close_point', default=20, type=int)
parser.add_argument('--dataset', default='cifar20', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=0.9, type=float)
parser.add_argument('--noise_type', default='subclass', type=str)
parser.add_argument('--noise_rate', default=0.12, type=float, help='noise rate in sub-classes')
args = parser.parse_args()
print(args)


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True


def calculate_sklearn_tSNE(features):
    print(getTime(), "tSNE start...")
    tsne_first = TSNE(n_components=2, random_state=0, n_jobs=-1).fit_transform(features)
    tx, ty = tsne_first[:, 0], tsne_first[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    print(getTime(), "tSNE end", tx.shape, ty.shape)
    return tx, ty


def calculate_eucli_dis(feature_bank, guess_index, train_noisy_labels, close_point=20):
    with torch.no_grad():
        other_index = np.setdiff1d(np.arange(len(train_noisy_labels)), guess_index)
        guess_feature = feature_bank[guess_index]
        feature_bank = feature_bank[other_index]
        train_noisy_labels = train_noisy_labels[other_index]

        guess_feature = torch.tensor(guess_feature).cuda()
        feature_bank = torch.tensor(feature_bank).cuda()

        dis_arr = []
        for class_i in range(args.num_classes):
            class_index = np.where(train_noisy_labels == class_i)[0]
            class_feature_bank = feature_bank[class_index]

            eucli_dis_matrix = torch.cdist(guess_feature, class_feature_bank)
            close_dis, _ = eucli_dis_matrix.topk(k=close_point, dim=-1, largest=False)
            close_mean_dis = torch.mean(close_dis, dim=1)

            top_dis, _ = close_mean_dis.topk(k=close_point, dim=-1, largest=False)

            distance = torch.mean(top_dis).cpu().item()
            dis_arr.append(round(distance, 2))

        return np.argsort(dis_arr)[0]


def scan_correct_subclass(X, eps, min_samples, features, train_noisy_labels, close_point=20):
    confident_index = np.array([])
    correct_train_labels = np.array(train_noisy_labels, copy=True)

    for class_i in range(args.num_classes):
        class_index = np.where(train_noisy_labels == class_i)[0]
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X[class_index])
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        dbscan_class_num_arr = []
        if n_clusters_ > 0:
            for i in range(n_clusters_):
                dbscan_class_num_arr.append(np.where(labels == i)[0].shape[0])
        else:
            print("class", class_i, " n_clusters_ 0 error!!!")
            confident_index = np.hstack((confident_index, class_index))
            continue

        largest_index = np.argmax(dbscan_class_num_arr)
        lar_exam_index = np.where(labels == largest_index)[0]
        confid_index = class_index[lar_exam_index]
        confident_index = np.hstack((confident_index, confid_index))

        # eucli dis
        for i in range(len(dbscan_class_num_arr)):
            if i != largest_index:
                guess_index = np.where(labels == i)[0]
                pred_class = calculate_eucli_dis(features, class_index[guess_index], train_noisy_labels, close_point)

                correct_train_labels[class_index[guess_index]] = pred_class
                confident_index = np.hstack((confident_index, class_index[guess_index]))

    unconfident_index = np.setdiff1d(np.arange(train_noisy_labels.shape[0]), confident_index)
    return confident_index.astype(int), unconfident_index.astype(int), correct_train_labels


def create_model(arch, num_classes=10):
    if arch == "resnet18":
        return ResNet18(num_classes).cuda()
    elif arch == "resnet34":
        return ResNet34(num_classes).cuda()
    else:
        return PreActResNet18(num_classes).cuda()

# prepare data
args.num_classes = 20
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)
test_set.targets = gen_subclass_noise(test_set.targets, 0)

train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, _ = getNoisyData(args.seed, args.dataset, args.data_path, args.data_percent, args.noise_type, args.noise_rate)
train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False,)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

model = create_model(args.arch, args.num_classes)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)

best_test_acc = 0
best_val_acc = 0
for epoch in range(args.num_epochs):
    if epoch < args.T1:
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    elif epoch == args.T1:
        print("\nNoiseCluter...")
        predict_dataset = Train_Dataset(train_data, train_noisy_labels, transform_test)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        fc = model.linear
        model.linear = nn.Identity()
        features = predict_repre(predict_loader, model)
        model.linear = fc
        tx, ty = calculate_sklearn_tSNE(features)

        confident_index, unconfident_index, train_noisy_labels = scan_correct_subclass(np.vstack((tx, ty)).T, args.eps, args.min_samples, features, train_noisy_labels, args.close_point)
        print(confident_index.shape, unconfident_index.shape, train_noisy_labels.shape)

        # Prepare corrected confident data
        train_dataset = Train_Dataset(train_data[confident_index], train_noisy_labels[confident_index], transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        # Loss function
        train_nums = np.zeros(args.num_classes, dtype=int)
        for item in train_noisy_labels[confident_index]:
            train_nums[item] += 1
        class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums).cuda()
        train_criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    else:
        train_loss, train_acc = train(model, train_loader, optimizer, train_criterion, epoch)

    val_loss, val_acc = evaluate(model, val_loader, criterion, "Val Acc:")
    scheduler.step()
    if best_val_acc < val_acc:
        test_loss, test_acc = evaluate(model, test_loader, criterion, "Epoch " + str(epoch + 1) + " Test Acc:")
        best_test_acc = test_acc
        best_val_acc = val_acc

print(getTime(), "Best Test Acc:", best_test_acc)
