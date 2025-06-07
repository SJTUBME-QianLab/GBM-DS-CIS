# encoding: utf-8
import argparse
import os
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import random
from tqdm import tqdm
from util import prf, get_train_and_test_datasets, get_batch_data, datatosim
from sig_cnn import ResSpiNet, ContrastiveLoss, SinkhornDistance, ParwiseLoss
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='PyTorch GBM Training')
parser.add_argument('--data', default='./input', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--outf', default='./output',
                    help='folder to output model checkpoints')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train', default=True,
                    help='train the model')
parser.add_argument('--test', action='store_true', default=True,
                    help='test a [pre]trained model on new images')
parser.add_argument('-t', '--fine-tuning', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')

def train(spiral_train_data, spiral_sim_imgs, sin_train_data, sin_sim_imgs, model,
          criterion, t_criterion, cont_criterion, paw_criterion, optimizer, epoch):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model = model.cuda()
    model.train()

    end = time.time()
    for batch_idx in range(20):
        input_spiral, target_spiral, sim_data_spiral, data_list = get_batch_data(spiral_train_data, spiral_sim_imgs, 4, 'spiral', [])
        input_sin, target_sin, sim_data_sin, _ = get_batch_data(sin_train_data, sin_sim_imgs, 4, 'sin', data_list)
        # measure data loading time
        data_time.update(time.time() - end)
        input = datatosim(input_spiral, input_sin, sim_data_spiral, sim_data_sin)
        target = torch.cat([target_spiral, target_sin], dim=0)
        L = len(target)

        if cuda:
            input, target = input.cuda(), target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            input_var = input_var.unsqueeze(1)
            input_var[input_var < 0] = 0
            input_var[input_var > 1] = 1
            output, out_features, actfeat, back_feat, cla_feat, cla_sim_feat = model(input_var)

            acti_f = out_features[0:L]
            acti_sim_f = out_features[L:2*L]
            back_f = out_features[2*L:]
            spi_acti_feat = acti_f[0:4 * 9]
            sin_acti_feat = acti_f[4 * 9:]

            spi_acti_f = spi_acti_feat.view(4, 9, -1).permute(1, 0, 2)
            cont_loss1 = torch.mean(
                torch.stack([cont_criterion(spi_acti_f[i], spi_acti_f[j]) for i in range(9) for j in range(i + 1, 9)]))
            sin_acti_f = sin_acti_feat.view(4, 9, -1).permute(1, 0, 2)
            cont_loss2 = torch.mean(
                torch.stack([cont_criterion(sin_acti_f[i], sin_acti_f[j]) for i in range(9) for j in range(i + 1, 9)]))

            spi_acti_sim_feat = acti_sim_f[0:4 * 9]
            sin_acti_sim_feat = acti_sim_f[4 * 9:]

            spi_acti_sim_f = spi_acti_sim_feat.view(4, 9, -1).permute(1, 0, 2)
            sin_acti_sim_f = sin_acti_sim_feat.view(4, 9, -1).permute(1, 0, 2)
            cont_loss4 = torch.mean(
                torch.stack([cont_criterion(spi_acti_sim_f[i], spi_acti_sim_f[j]) for i in range(9) for j in range(i + 1, 9)]))
            cont_loss5 = torch.mean(
                torch.stack(
                    [cont_criterion(sin_acti_sim_f[i], sin_acti_sim_f[j]) for i in range(9) for j in range(i + 1, 9)]))

            cont_loss3 = torch.mean(
                torch.stack([cont_criterion(sin_acti_f[i], spi_acti_f[j]) for i in range(9) for j in range(9)]))
            t_loss = torch.mean(torch.abs(
                F.cosine_similarity(acti_f.view(L, -1).unsqueeze(1), back_f.view(L, -1).unsqueeze(0), dim=-1) - 0.4))
            cla_loss0 = criterion(output, target_var.to(torch.int64))
            cla_loss1 = criterion(cla_feat, target_var.to(torch.int64))
            cla_sim_loss1 = criterion(cla_sim_feat, target_var.to(torch.int64))
            cs_loss1 = -torch.mean(F.cosine_similarity(actfeat[1].view(L, -1), back_feat.view(L, -1))) + 0.1
            cs_loss = torch.mean(torch.abs(torch.softmax(actfeat[1], dim=1) - 0.5))

            loss = cla_loss0 + 0.3*(cla_loss1 + cla_sim_loss1) + 1.5*(cont_loss1 + cont_loss2 + cont_loss3 + cont_loss4 + cont_loss5) + \
                   t_loss + cs_loss + 0.01*cs_loss1
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss, input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    print('Epoch: [{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
          'Prec@1 {top1_val:.4f} ({top1_avg:.4f})'.format(
        epoch, batch_time=batch_time,
        data_time=data_time,
        loss_val =losses.val.detach().cpu().numpy().item(),
        loss_avg = losses.avg.detach().cpu().numpy().item(),
        top1_val=top1.val.detach().cpu().numpy().item(),
        top1_avg=top1.avg.detach().cpu().numpy().item()))
    return top1.val.detach().cpu().numpy().item(), losses.avg.detach().cpu().numpy().item(), \
           [cla_loss0, cla_loss1, cont_loss1, cont_loss2, cont_loss3, t_loss, cs_loss]

def test(test_loader, model):
    model.eval()
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    counter = 0
    # Evaluate all the validation set
    y_true = []
    y_pre = []
    y_scores = []
    test_data = [temp[0] for temp in test_loader[0]]
    test_target = [temp[1] for temp in test_loader[0]]
    test_sim_data = test_loader[1]

    spi_test = test_data[0:360]
    spi_target = test_target[0:360]
    spi_sim_data = test_sim_data[0:360]
    sin_test = test_data[360:]
    sin_target = test_target[360:]
    sin_sim_data = test_sim_data[360:]
    sin_test = [sin_test[i] for i in range(len(sin_test)) if (i+1) % 10 != 0]
    sin_target = [sin_target[i] for i in range(len(sin_target)) if (i+1) % 10 != 0]
    sin_sim_data = torch.stack([sin_sim_data[i] for i in range(len(sin_sim_data)) if (i+1) % 10 != 0], dim=0)
    test_sort = torch.randperm(len(spi_test)).numpy().tolist()

    for i in range(9):
        temp_test_index = test_sort[40 * i:40 * (i + 1)]
        spi_input = np.array(spi_test)[temp_test_index]
        spi_label = np.array(spi_target)[temp_test_index]
        sin_input = np.array(sin_test)[temp_test_index]
        sin_lable = np.array(sin_target)[temp_test_index]

        spi_sim_img = spi_sim_data[temp_test_index]
        sin_sim_img = sin_sim_data[temp_test_index]

        spi_input = np.vstack([spi_input, sin_input])
        spi_label = np.hstack([spi_label, sin_lable])

        input = torch.from_numpy(np.stack(np.array(spi_input)))
        target = torch.from_numpy(np.stack(np.array(spi_label)))
        sim_img = torch.cat([spi_sim_img, sin_sim_img]).view(80, 224, 224)

        diff_input = input - sim_img
        datalist = list(range(len(diff_input)))
        shuffled_indices = random.sample(datalist, len(datalist))
        shuffled_matrix_spiral = np.maximum(diff_input + sim_img[shuffled_indices], 0)
        input = torch.cat([input, shuffled_matrix_spiral, sim_img])

        if cuda:
            input = torch.Tensor(input).cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                # compute output
                input_var = input_var.unsqueeze(1)  # 扩展维度
                input_var[input_var < 0] = 0
                input_var[input_var > 1] = 1
                output, _, _, _, _,_ = model(input_var)
            # Take last layer output
            if isinstance(output, tuple):
                output = output[len(output) - 1]
            predlabel = torch.from_numpy(output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1))
            counter += torch.sum(predlabel.eq(target)).numpy()
            y_true.extend(target.numpy().tolist())
            y_pre.extend(predlabel.numpy().tolist())
            y_score = F.softmax(output, dim=1).detach().cpu().numpy()
            y_scores.extend(y_score[:, 1].tolist())
    P, R, F1, ACC, TPR, TNR, AUC = prf(y_true, y_pre, y_scores)
    return P, R, F1, ACC, TPR, TNR, AUC

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    test_best_acc = 0
    print("=> using cuda: {cuda}".format(cuda=cuda))
    model = ResSpiNet(name='resnet18')
    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss()
    t_criterion = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
    paw_criterion = ParwiseLoss()
    cont_criterion = ContrastiveLoss()
    if cuda:
        criterion.cuda()
        t_criterion.cuda()
        cont_criterion.cuda()
        sinkhorn.cuda()
        paw_criterion.cuda()

    # Set SGD + Momentum
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    schedulers = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    ############ TRAIN/EVAL/TEST ############
    cudnn.benchmark = True
    P, R, F1, ACC, TPR, TNR, AUC = 0, 0, 0, 0, 0, 0, 0
    train_loss  = []
    train_pre = []
    cla_loss0 = []
    cla_loss1 = []
    cont_loss1 = []
    cont_loss2 = []
    cont_loss3 = []
    t_loss = []
    cs_loss = []
    if args.train:
        print("=> training...")
        for epoch in range(args.start_epoch, args.epochs):
            schedulers.step()
            # Train for one epoch
            train_acc, loss, all_loss = train(spiral_train_data, spiral_sim_imgs, sin_train_data, sin_sim_imgs,
                                    model, criterion, t_criterion, cont_criterion, paw_criterion, optimizer, epoch)
            train_loss.append(loss)
            train_pre.append(train_acc)
            # Evaluate on validation set
            model = model.eval()
            tP, tR, tF1, tACC, tTPR, tTNR, tAUC = test(test_loader, model)
            is_test_best = bool(tACC >= test_best_acc)

            if is_test_best:
                print("=> best testing and save model...")
                print(
                    'train_acc:{:.4f}, test_prec1:{:.4f}, best_test_prec1:{:.4f}'.format(
                        train_acc, tACC, test_best_acc))
                test_best_acc = max(tACC, test_best_acc)
                torch.save(model,
                           '/home/data2/sjtu/checkpoint/Sig_CIM/repeat/model-{}-time-{}-fold.pth'.format(irun, fold))
                P, R, F1, ACC, TPR, TNR, AUC = tP, tR, tF1, tACC, tTPR, tTNR, tAUC
            model = model.train()
            cla_loss0.append(all_loss[0].detach().cpu().numpy())
            cla_loss1.append(all_loss[1].detach().cpu().numpy())
            cont_loss1.append(all_loss[2].detach().cpu().numpy())
            cont_loss2.append(all_loss[3].detach().cpu().numpy())
            cont_loss3.append(all_loss[4].detach().cpu().numpy())
            t_loss.append(all_loss[5].detach().cpu().numpy())
            cs_loss.append(all_loss[6].detach().cpu().numpy())
    return train_acc, P, R, F1, ACC, TPR, TNR, AUC

if __name__ == '__main__':
    run = 10
    ifolds = 4
    acc = np.zeros((run, ifolds), dtype=float)
    precision = np.zeros((run, ifolds), dtype=float)
    recall = np.zeros((run, ifolds), dtype=float)
    f_score = np.zeros((run, ifolds), dtype=float)
    auc = np.zeros((run, ifolds), dtype=float)
    tpr = np.zeros((run, ifolds), dtype=float)
    tnr = np.zeros((run, ifolds), dtype=float)

    for irun in tqdm(range(run)):
        for fold in range(ifolds):
            spiral_datasets, sin_dataset = get_train_and_test_datasets(irun, fold)
            spiral_train_data, spiral_sim_imgs, spiral_test_data, spiral_sim_imgs_test = spiral_datasets
            sin_train_data, sin_sim_imgs, sin_test_data, sin_sim_imgs_test = sin_dataset
            spiral_test_data.extend(sin_test_data)
            test_datasets = spiral_test_data
            sim_imgs_test = torch.cat((spiral_sim_imgs_test, sin_sim_imgs_test))
            test_sort = torch.randperm(len(test_datasets)).numpy().tolist()
            test_loader = [test_datasets, sim_imgs_test, test_sort]

            args = parser.parse_args()
            cuda = torch.cuda.is_available()
            train_acc, P, R, F1, ACC, TPR, TNR, AUC = main()
            acc[irun][fold], recall[irun][fold], precision[irun][fold], f_score[irun][
                fold], auc[irun][fold], tpr[irun][fold], tnr[irun][
                fold] = ACC, R, P, F1, AUC, TPR, TNR

            print("irun =", irun)
            print("fold=", fold)
            print('mi-net mean accuracy = ', np.mean(acc))
            print('std = ', np.std(acc))
            print('mi-net mean precision = ', np.mean(precision))
            print('std = ', np.std(precision))
            print('mi-net mean recall = ', np.mean(recall))
            print('std = ', np.std(recall))
            print('mi-net mean fscore = ', np.mean(f_score))
            print('std = ', np.std(f_score))
            print('mi-net mean auc = ', np.mean(auc))
            print('std = ', np.std(auc))
            print('mi-net mean tpr = ', np.mean(tpr))
            print('std = ', np.std(tpr))
            print('mi-net mean tnr = ', np.mean(tnr))
            print('std = ', np.std(tnr))


