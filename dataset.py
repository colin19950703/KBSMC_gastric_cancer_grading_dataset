from glob import glob
from PIL import Image
from collections import Counter
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import dataset


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def print_number_of_sample(data_set, prefix):
    def fill_empty_label(label_dict):
        for i in range(max(label_dict.keys()) + 1):
            if label_dict[i] != 0:
                continue
            else:
                label_dict[i] = 0
        return dict(sorted(label_dict.items()))

    data_label = [data_set[i][1] for i in range(len(data_set))]
    d = Counter(data_label)
    d = fill_empty_label(d)
    print("%-7s" % prefix, d)
    data_label = [d[key] for key in d.keys()]

    return data_label

def prepare_gastric_data(data_root_dir='./KBSMC_Gastric_WSI_Cancer_Grading_1024/', nr_classes=4):
    def load_data_info_from_list(wsi_list, data_root_dir, gt_list, nr_claases):
        file_list = []
        for wsi_name in wsi_list:
            pathname = glob(f'{data_root_dir}/{wsi_name}/*.jpg')
            file_list.extend(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))

        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_claases]
        return list_out

    gt_list = {1: 0,  # "BN", #0
               2: 0,  # "BN", #0
               3: 1,  # "TW", #2
               4: 2,  # "TM", #3
               5: 3,  # "TP", #4
               }

    WSI_dir = data_root_dir + '/WSIs'
    csv_path = data_root_dir + '/WSIs_Split_Info.csv'

    df = pd.read_csv(csv_path).iloc[:, :3]
    train_list = list(df.query('Task == "train"')['WSI'])
    valid_list = list(df.query('Task == "val"')['WSI'])
    test_list = list(df.query('Task == "test"')['WSI'])


    train_set = load_data_info_from_list(train_list, WSI_dir, gt_list, nr_classes)
    valid_set = load_data_info_from_list(valid_list, WSI_dir, gt_list, nr_classes)
    test_set = load_data_info_from_list(test_list, WSI_dir, gt_list, nr_classes)


    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test')

    return train_set, valid_set, test_set

class DatasetSerial(data.Dataset):
    """get image by index
    """

    def __init__(self, pair_list, img_transform=None, target_transform=None, two_crop=False):
        self.pair_list = pair_list

        self.img_transform = img_transform
        self.target_transform = target_transform
        self.num = self.__len__()

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.img_transform is not None:
            img = self.img_transform(image)
        else:
            img = image

        return img, target

def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size

if __name__ == '__main__':
    print('\nGastric')
    train, valid, test = prepare_gastric_data(data_root_dir='./KBSMC_gastric_cancer_grading_512/')
    train_dataset = dataset.DatasetSerial(train)
    visualize(train_dataset, 5)
