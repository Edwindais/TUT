import torch
import numpy as np
import os
import random
import math
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate


def my_collate_func(batchs):
    features = [item['feature'] for item in batchs]
    labels = [item['label'] for item in batchs]
    vids = [item['vid'] for item in batchs]
    lengths = [item['length'] for item in batchs]
    batch_size = len(batchs)
    max_seq_length = max(lengths)  # max length in one batch

    mask_batch = torch.zeros((batch_size, max_seq_length), dtype=torch.float)
    fea_batch = []
    label_batch = []
    for i in range(batch_size):
        fea_batch.append(torch.from_numpy(features[i].T))
        label_batch.append(torch.from_numpy(labels[i]))
        mask_batch[i, :lengths[i]] = 1

    return  dict(
        feature = pad_sequence(fea_batch, batch_first=True, padding_value = 0.).transpose(1, 2),  # (B, D, L)
        label = pad_sequence(label_batch, batch_first=True, padding_value = -100).long(),  # (B, L)
        vid = vids,
        length = lengths,
        mask = mask_batch,  # (B, L)
        )


class Dataset_food(Dataset):
    def __init__(self, root="/mnt/hdd2/Projects_dai/Projects_dai/ASFormer/data/", dataset="50salads", split="1", mode="train"):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.list_of_examples = []
        self.actions_dict = {}
        self.reverse_dict = {}

        # use the full temporal resolution @ 15fps
        self.sample_rate = 1
        # sample input features @ 15fps instead of 30 fps
        # for 50salads, and up-sample the output to 30 fps
        if dataset == "50salads":
            self.sample_rate = 2

        self.vid_list_file = root+dataset+"/splits/"+mode+".split"+split+".bundle"
        self.features_path = root+dataset+"/features/"
        self.gt_path = root+dataset+"/groundtruth/"
        self.__read_mapping__()
        self.__read_data__()
        self.num_classes = len(self.actions_dict)  # class num of the dataset
        # self.__show_info__()

    def __read_mapping__(self):
        mapping_file = self.root+ self.dataset + "/mapping.txt"
        with open(mapping_file, "r", encoding="utf-8") as f:
            actions = [ln.strip() for ln in f if ln.strip()]   # keep all non‑empty lines
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
            self.reverse_dict[int(a.split()[0])] = a.split()[1]

    def __read_data__(self):
        with open(self.vid_list_file, 'r') as f:
            self.list_of_examples = [line.strip() for line in f if line.strip()]
    
    def __show_info__(self):
        print("action dict:  ", self.actions_dict)
        print("list_of_examples:  ", self.list_of_examples)
        print("num_classes:  ", self.num_classes)
    
    def __get_actions_dict__(self):
        return self.actions_dict
    
    def __get_sample_rate__(self):
        return self.sample_rate

    def __getitem__(self, index):
        vid = self.list_of_examples[index]
        feat_base = self.features_path + vid.split('.')[0]
        if os.path.exists(feat_base + '.npy'):
            features = np.load(feat_base + '.npy')
        elif os.path.exists(feat_base + '.pth'):
            tmp = torch.load(feat_base + '.pth')
            if isinstance(tmp, torch.Tensor):
                features = tmp.cpu().numpy()
            else:
                features = np.array(tmp)
        else:
            raise FileNotFoundError(f"No feature file found for video: {vid}")
        # print(features.shape)  # (D, L)

        with open(self.gt_path + vid + ".txt", "r", encoding="utf-8") as f:
            content = [ln.strip() for ln in f if ln.strip()]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # print(classes.shape)  # (L)

        features_down = features[:, ::self.sample_rate]
        classes_down = classes[::self.sample_rate]
        vlength = np.shape(features_down)[1]

        # (D, L), (L), [video id], [video length]
        return {
            "feature": features_down,
            "label": classes_down,
            "vid": vid,
            "length": vlength
        }

    def __len__(self):
        return len(self.list_of_examples)


class Dataset_toy(Dataset):
    def __init__(self, root="data/", dataset="50salads", split="1", mode="train"):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.list_of_examples = []
        self.actions_dict = {}
        self.reverse_dict = {}
        self.sample_rate = 1
        if dataset == "50salads":
            self.sample_rate = 2
        self.vid_list_file = root+dataset+"/splits/"+mode+".split"+split+".bundle"
        # if mode == test
        self.features_path = root+dataset+"/features/"
        self.gt_path = root+dataset+"/groundTruth/"
        self.__read_mapping__()
        self.__read_data__()
        self.num_classes = len(self.actions_dict)  # class num of the dataset
        # self.__show_info__()

    def __read_mapping__(self):
        mapping_file = self.root+self.dataset+"/mapping.txt"
        file_ptr = open(mapping_file, 'r')
        with open(mapping_file, "r", encoding="utf-8") as f:
            actions = [ln.strip() for ln in f if ln.strip()]   # 既去掉空串，又保留末尾数据
        file_ptr.close()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
            self.reverse_dict[int(a.split()[0])] = a.split()[1]

    def __read_data__(self):
        with open(self.vid_list_file, "r", encoding="utf-8") as f:
            self.list_of_examples = [ln.strip() for ln in f if ln.strip()]
    
    def __show_info__(self):
        print("action dict:  ", self.actions_dict)
        print("list_of_examples:  ", self.list_of_examples)
        print("num_classes:  ", self.num_classes)
    
    def __get_actions_dict__(self):
        return self.actions_dict
    
    def __get_sample_rate__(self):
        return self.sample_rate

    def __getitem__(self, index):
        vid = self.list_of_examples[index]
        feat_base = self.features_path + vid.split('.')[0]
        if os.path.exists(feat_base + '.npy'):
            features = np.load(feat_base + '.npy')
        elif os.path.exists(feat_base + '.pth'):
            tmp = torch.load(feat_base + '.pth')
            if isinstance(tmp, torch.Tensor):
                features = tmp.cpu().numpy()
            else:
                features = np.array(tmp)
        else:
            raise FileNotFoundError(f"No feature file found for video: {vid}")
        # print(features.shape)  # (D, L)

        with open(self.gt_path + vid, "r", encoding="utf-8") as f:
            content = [ln.strip() for ln in f if ln.strip()]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # print(classes.shape)  # (L)

        features_down = features[:, ::self.sample_rate]
        classes_down = classes[::self.sample_rate]
        vlength = np.shape(features_down)[1]

        # (D, L), (L), [video id], [video length]
        return {
            "feature": features_down,
            "label": classes_down,
            "vid": vid,
            "length": vlength
        }

    def __len__(self):
        return len(self.list_of_examples)


if __name__ == '__main__':
    mydataset = Dataset_food(root='/mnt/hdd2/Projects_dai/Projects_dai/ASFormer/data/', dataset="surgery_I3D_new")
    print(mydataset[5])
    train_loader = torch.utils.data.DataLoader(
        mydataset,
        batch_size=8,
        num_workers=4,
        collate_fn=my_collate_func
    )
    for i, d in enumerate(train_loader):
        print(d['feature'].shape)
        print(d['label'].shape)
        print(d['vid'])
        print(d['length'])
        print(d['mask'].shape)
