# encoding: utf-8
import os
import numpy as np
from sklearn import metrics
import torch
import random
import pickle
from PIL import Image

def prf(y_true, y_pred, y_score):

    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    TPR = Recall  # 灵敏度
    TNR = TN / (FP + TN) if (FP + TN) != 0. else TN
    AUC = metrics.roc_auc_score(y_true, y_score)

    return Precision, Recall, F1_score, Accuracy, TPR, TNR, AUC

def comput_similar_img(datasets, back_imgs):
    Ld = len(datasets)
    Lb = len(back_imgs)
    data_img = [data[0] for data in datasets]
    newdata_img = data_img
    newback_imgs = back_imgs
    newdata_img = torch.Tensor(np.array(newdata_img)).view(Ld, -1)
    newback_imgs = torch.Tensor(np.array(newback_imgs)).view(Lb, -1)
    sim_img_list = []
    for i in range(Ld):
        single_sim = [torch.cosine_similarity(newback_imgs[j], newdata_img[i], dim=0) for j in range(Lb)]
        single_l = torch.max(torch.stack(single_sim), 0)[1]
        sim_img_list.append(single_l)
    return newback_imgs[sim_img_list]

def get_train_and_test_datasets(irun, fold):
    spiral_box = (60, 100, 515, 630)
    spiral_back_dataset = pickle.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/gen_dataset/all_spiral_back_dataset.pkl', 'rb'))

    spiral_dataset = pickle.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/gen_dataset/all_spiral_dataset-{}time-{}fold.pkl'.format(irun, fold), 'rb'))
    spiral_train_datasets = spiral_dataset[0]
    spiral_test_datasets = spiral_dataset[1]

    spiral_train_data = [[np.array(Image.fromarray(temp_d[0]).crop(spiral_box).resize((224, 224))), temp_d[1]]
                         for temp_d in spiral_train_datasets]
    spiral_test_data = [[np.array(Image.fromarray(temp_d[0]).crop(spiral_box).resize((224, 224))), temp_d[1]]
                        for temp_d in spiral_test_datasets]

    spiral_back_dataset = [np.array(Image.fromarray(temp_d).crop(spiral_box).resize((224, 224)))
                           for temp_d in spiral_back_dataset]

    if os.path.isfile('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold)):
        spiral_sim_imgs = torch.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        spiral_sim_imgs = comput_similar_img(spiral_train_data, spiral_back_dataset)
        torch.save(spiral_sim_imgs,
                   'Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold))

    if os.path.isfile('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold)):
        spiral_sim_imgs_test = torch.load(
            open('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(
                irun, fold), 'rb'))
    else:
        spiral_sim_imgs_test = comput_similar_img(spiral_test_data, spiral_back_dataset)
        torch.save(spiral_sim_imgs_test,
                   'Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold))

    #################################################################################################
    sin_box = (25, 20, 175, 220)
    sin_back_dataset = pickle.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/gen_dataset/all_sin_back_dataset.pkl', 'rb'))

    sin_dataset = pickle.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/gen_dataset/all_sin_dataset-{}time-{}fold.pkl'.format(irun, fold), 'rb'))
    sin_train_datasets = sin_dataset[0]
    sin_test_datasets = sin_dataset[1]

    sin_train_data = [[np.array(Image.fromarray(temp_d[0]).crop(sin_box).resize((224, 224))), temp_d[1]]
                      for temp_d in sin_train_datasets]
    sin_test_data = [[np.array(Image.fromarray(temp_d[0]).crop(sin_box).resize((224, 224))), temp_d[1]]
                     for temp_d in sin_test_datasets]

    sin_back_dataset = [np.array(Image.fromarray(temp_d).crop(sin_box).resize((224, 224)))
                        for temp_d in sin_back_dataset]

    if os.path.isfile('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sin_sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold)):
        sin_sim_imgs = torch.load(open('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sin_sim_imgs-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        sin_sim_imgs = comput_similar_img(sin_train_data, sin_back_dataset)
        torch.save(sin_sim_imgs,
                   'Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sin_sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold))

    if os.path.isfile('save/resnet18_spiral_transfer_cont/sin_sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold)):
        sin_sim_imgs_test = torch.load(
            open('Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sin_sim_imgs_test-{}-time-{}-fold.pkl'.format(
                irun, fold), 'rb'))
    else:
        sin_sim_imgs_test = comput_similar_img(sin_test_data, sin_back_dataset)
        torch.save(sin_sim_imgs_test,
                   'Dataset/ST_CEM_and_Sig_CIM_Dataset/ST_CEM_and_Sig_CIM_Dataset/resnet18_spiral_transfer_cont/sin_sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold))

    return [spiral_train_data, spiral_sim_imgs, spiral_test_data, spiral_sim_imgs_test], \
           [sin_train_data, sin_sim_imgs, sin_test_data, sin_sim_imgs_test]

def get_batch_data(train_loader, back_dataset, batchsize, model, data_list):

    if model == 'spiral':
        times = 9
    elif model == 'sin':
        times = 10

    if len(data_list) == 0:
        num_0 = int(len([0 for temp in train_loader if temp[1] == 0])/times)
        num_1 = int(len([1 for temp in train_loader if temp[1] == 1])/times)
        psp_d = random.sample(list(range(num_0)), int(batchsize/2))
        ttp_d = random.sample(list(range(num_1)), int(batchsize/2))
        psp_list = [temp_psp * times + i for temp_psp in psp_d for i in range(9)]
        ttp_list = [(temp_ttp+num_0) * times + i for temp_ttp in ttp_d for i in range(9)]
        data_list = psp_list + ttp_list

    data = [train_loader[dx][0] for dx in data_list]
    label = [train_loader[dx][1] for dx in data_list]
    sim_data = torch.cat([back_dataset[dx] for dx in data_list]).view(-1, 224, 224)
    return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label)), sim_data, data_list


def datatosim(input_spiral, input_sin, sim_data_spiral, sim_data_sin):
    diff_spiral = input_spiral - sim_data_spiral
    diff_sin = input_sin - sim_data_sin
    datalist = list(range(len(input_spiral)))
    shuffled_indices = random.sample(datalist, len(datalist))
    shuffled_matrix_spiral = np.maximum(diff_spiral + sim_data_spiral[shuffled_indices], 0)
    shuffled_matrix_sin = np.maximum(diff_sin + sim_data_sin[shuffled_indices], 0)
    input = torch.cat([input_spiral, input_sin, shuffled_matrix_spiral, shuffled_matrix_sin, sim_data_spiral, sim_data_sin])
    return input