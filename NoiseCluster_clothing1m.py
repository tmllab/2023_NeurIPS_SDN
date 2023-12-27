import os
import os.path
import argparse
import random
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torch.backends.cudnn as cudnn
from sklearn.cluster import DBSCAN
from MulticoreTSNE import MulticoreTSNE as TSNE

from common.tools import AverageMeter, getTime, evaluate, predict_softmax, accuracy, ProgressMeter, Clothing1M_Dataset, predict_repre, Clothing1M_Unlabeled_Dataset


parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--data_root', type=str, default='data/Clothing1M_Official/')
parser.add_argument('--data_percent', default=0.95, type=float, help='T1')
parser.add_argument('--pretrain', action='store_false', help='pretrain')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=0.001, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iters_epoch', default=10, type=int)
parser.add_argument('--eps', default=0.04, type=float)
parser.add_argument('--min_samples', default=100, type=float)
parser.add_argument('--filter_num', action='store', type=int, nargs='*', default=[12])
args = parser.parse_args()
print(args)
os.system('mkdir -p %s' % ('logs'))
os.system('mkdir -p %s' % ('model'))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True


def create_model(pretrained):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048, args.num_classes)
    return model.cuda()


def train_by_iter(model, train_iter, ceriation, train_optimizer, num_iter):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_iter), [batch_time, losses, top1], prefix="Train ")
    end = time.time()
    for batch_idx in range(num_iter):
        try:
            images, labels = next(train_iter)
            images = images.cuda()
            labels = labels.cuda()
        except StopIteration:
            break

        train_optimizer.zero_grad()
        logits = model(images)
        loss = ceriation(logits, labels)

        loss.backward()
        train_optimizer.step()

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(batch_idx)
    return top1.avg, losses.avg


def update_trainloader(model, train_data, train_noisy_labels, val_nums):
    predict_dataset = Clothing1M_Unlabeled_Dataset(train_data, args.data_root, train_transform)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model)
    probs, preds = torch.max(soft_outs.data, 1)

    confident_indexs = []
    unconfident_indexs = []
    for i in range(len(train_noisy_labels)):
        if preds[i] == train_noisy_labels[i]:
            confident_indexs.append(i)
        else:
            unconfident_indexs.append(i)

    print(getTime(), "confident and unconfident num:", len(confident_indexs), len(unconfident_indexs))
    confident_index = np.array(confident_indexs)
    unconfident_index = np.array(unconfident_indexs)

    # Loss function
    train_nums = np.zeros(args.num_classes, dtype=int)
    for item in preds[confident_index]:
        train_nums[item] += 1
    class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).cuda()

    return confident_index, unconfident_index, class_weights


def scan_correct_subclass_filter(X, eps, min_samples, train_noisy_labels, filter_num):
    confident_index = np.array([])
    correct_train_labels = np.array(train_noisy_labels, copy=True)

    for class_i in range(args.num_classes):
        class_index = np.where(train_noisy_labels == class_i)[0]
        if class_i in filter_num:
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
                    pred_class = calculate_eucli_dis(features, class_index[guess_index], train_noisy_labels)

                    correct_train_labels[class_index[guess_index]] = pred_class
                    confident_index = np.hstack((confident_index, class_index[guess_index]))
        else:
            confident_index = np.hstack((confident_index, class_index))

    return confident_index.astype(int), correct_train_labels


def calculate_eucli_dis(feature_bank, guess_index, train_noisy_labels, top_min_point=20, close_point=20):
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
            top_dis, _ = close_mean_dis.topk(k=top_min_point, dim=-1, largest=False)
            distance = torch.mean(top_dis).cpu().item()
            dis_arr.append(round(distance, 2))

        return np.argsort(dis_arr)[0]


def calculate_Multicore_tSNE(features, n_jobs=8):
    print(getTime(), "tSNE start...")
    t1 = datetime.datetime.now()

    tsne_first = TSNE(n_jobs=n_jobs).fit_transform(features)
    tx, ty = tsne_first[:, 0], tsne_first[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    delta = round((datetime.datetime.now() - t1).total_seconds() / 60, 1)
    print(getTime(), f"Used {delta} min,", "tSNE end", tx.shape, ty.shape)
    return tx, ty


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
])
args.num_classes = 14

# Load data file
kvDic = np.load(args.data_root + 'Clothing1m-data.npy', allow_pickle=True).item()
val_data = kvDic['clean_val_data']
val_labels = kvDic['clean_val_labels']
val_nums = np.zeros(args.num_classes, dtype=int)
for item in val_labels:
    val_nums[item] += 1
val_dataset = Clothing1M_Dataset(val_data, val_labels, args.data_root, transform_test)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, num_workers=4, pin_memory=True, shuffle=False, drop_last=False)

test_data = kvDic['test_data']
test_labels = kvDic['test_labels']
test_dataset = Clothing1M_Dataset(test_data, test_labels, args.data_root, transform_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size * 2, num_workers=4, pin_memory=True, shuffle=False, drop_last=False)

original_train_data = kvDic['train_data']
original_train_labels = kvDic['train_labels']
shuffle_index = np.arange(len(original_train_labels), dtype=int)
np.random.shuffle(shuffle_index)
original_train_data = original_train_data[shuffle_index]
original_train_labels = original_train_labels[shuffle_index]

# Prepare new data loader
nosie_len = int(len(original_train_labels) * args.data_percent)
whole_train_data = original_train_data[:nosie_len]
whole_train_labels = original_train_labels[:nosie_len]
train_dataset = Clothing1M_Dataset(whole_train_data, whole_train_labels, args.data_root, train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

model = create_model(args.pretrain)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_iters_epoch, args.lr / 100)
train_nums = np.zeros(args.num_classes, dtype=int)
for item in whole_train_labels:
    train_nums[item] += 1
class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

step = 0
train_iter = iter(train_loader)
num_iter = int((len(train_iter) - 1) / args.num_iters_epoch)
for iter_index in range(args.num_iters_epoch):
    step += 1
    train_acc, train_loss = train_by_iter(model, train_iter, criterion, optimizer, num_iter)
    val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), "Step " + str(step) + ", Val Acc:")
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), "Step " + str(step) + ", Test Acc:")
    scheduler.step()

# Extract confident data with early stopping
rest_train_data = original_train_data[nosie_len:]
rest_train_labels = original_train_labels[nosie_len:]
confident_index, unconfident_index, class_weights = update_trainloader(model, rest_train_data, rest_train_labels, val_nums)
print(confident_index.shape, unconfident_index.shape, class_weights.shape)
predict_dataset = Clothing1M_Dataset(rest_train_data[confident_index], rest_train_labels[confident_index], args.data_root, transform_test)
predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False, drop_last=False)

# Extract features and gen tSNE
base_filepath = "./model/clothing_base_" + str(args.seed) + ".hdf5"
torch.save(model.state_dict(), base_filepath)
model.fc = nn.Identity()
features = predict_repre(predict_loader, model)
tx, ty = calculate_Multicore_tSNE(features, -1)

# NoiseCluter
db_con_index, db_con_labels = scan_correct_subclass_filter(np.vstack((tx, ty)).T, args.eps, args.min_samples, rest_train_labels[confident_index], args.filter_num)

# Prepare corrected confident data
re_train_data = rest_train_data[confident_index][db_con_index]
re_train_labels = db_con_labels[db_con_index]
re_dataset = Clothing1M_Dataset(re_train_data, re_train_labels, args.data_root, train_transform)
re_loader = DataLoader(dataset=re_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

# Continue to train
model = create_model(args.pretrain)
model.load_state_dict(torch.load(base_filepath))
optimizer = optim.SGD(model.parameters(), lr=args.lr / 10, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_iters_epoch)
train_nums = np.zeros(args.num_classes, dtype=int)
for item in re_train_labels:
    train_nums[item] += 1
class_weights = torch.FloatTensor(np.mean(train_nums) / train_nums * val_nums / np.mean(val_nums)).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

best_val_acc = 0
best_test_acc = 0
re_iter = iter(re_loader)
num_iter = int((len(re_iter) - 1) / args.num_iters_epoch)
for iter_index in range(args.num_iters_epoch):
    step += 1
    train_acc, train_loss = train_by_iter(model, re_iter, criterion, optimizer, num_iter)
    val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), "Step " + str(step) + ", Val Acc:")
    scheduler.step()

    if (val_acc > best_val_acc):
        test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), "Step " + str(step) + ", Test Acc:")
        best_val_acc = val_acc
        best_test_acc = test_acc

print(getTime(), "Best Test Acc:", best_test_acc)