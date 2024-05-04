from utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from imageio import imread
import random
# import train

global epoch_num
epoch_num = 0

def number_of_certain_probability(sequence, probability):  # 以一定概率随机选择一个数
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class myDataset(Dataset):
    def __init__(self, args, phase,layer_num,size):

        self.img = os.path.join(args.data_path, phase, "img_color")
        self.depth = os.path.join(args.data_path, phase, "depth")

        self.img_list = os.listdir(self.img)
        self.depth_list = os.listdir(self.depth)
        self.layer_num = layer_num
        self.size = size
        self.dataset_average = args.dataset_average

        layer_range = self.layer_num

        self.list_k = list(range(layer_range))
        self.probability = [1 / layer_range] * layer_range

        self.img_list.sort(key=lambda x: int(x.split('.')[0]))
        self.depth_list.sort(key=lambda x: int(x.split('.')[0]))

        

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        im_name = self.img_list[idx]
        img_path = os.path.join(self.img, im_name)
        depth_path = os.path.join(self.depth, im_name)
        flip = np.random.randint(low=0, high=100)
        flip = flip % 4

        img = imread(img_path)
        # if flip == 1:
        #     img = cv2.flip(img, flipCode=1)
        # if flip == 2:
        #     img = cv2.flip(img, flipCode=0)
        # if flip == 3:
        #     img = cv2.flip(img, flipCode=-1)
        if len(img.shape) < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img[..., 0, np.newaxis]
        im = im2float(img, dtype=np.float32)  # convert to double, max 1
        #sRGB to linear
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                       / 211) ** (12 / 5)
        amp = np.sqrt(im)  # to amplitude
        amp = np.transpose(amp, axes=(2, 0, 1))
        amp = resize_keep_aspect(amp, [self.size, self.size])
        amp = np.reshape(amp, (1, self.size, self.size))

        depth = imread(depth_path)
        # if flip == 1:
        #     depth = cv2.flip(depth, flipCode=1)
        # if flip == 2:
        #     depth = cv2.flip(depth, flipCode=0)
        # if flip == 3:
        #     depth = cv2.flip(depth, flipCode=-1)
        if len(depth.shape) < 3:
            depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        depth = depth[..., 1, np.newaxis]
        depth = im2float(depth, dtype=np.float32)
        depth = np.transpose(depth, axes=(2, 0, 1))
        depth = resize_keep_aspect(depth, [self.size, self.size])
        depth = np.reshape(depth, (1, self.size, self.size))
        depth = 1 - depth

        if self.dataset_average:
            ikk = number_of_certain_probability(self.list_k, self.probability)
        else:
            list_ikk = random.sample(self.list_k, 1)
            ikk = list_ikk[0]

        mask = np.logical_and(depth >= ikk / self.layer_num, depth < (ikk + 1) / self.layer_num)

        while mask.sum() == 0:
            if self.dataset_average:
                ikk = number_of_certain_probability(self.list_k, self.probability)
            else:
                list_ikk = random.sample(self.list_k, 1)
                ikk = list_ikk[0]
            mask = np.logical_and(depth >= ikk / self.layer_num, depth < (ikk + 1) / self.layer_num)


        return amp, depth, mask, ikk


def data_loader(args,type="train"):

    train_images = myDataset(args, type, args.layer_num,args.img_size)
    train_loader = DataLoader(
        train_images, batch_size=args.size_of_miniBatches, shuffle=True)

    return train_loader



