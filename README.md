# KBSMC_gastric_cancer_grading_dataset
This repository provides the KBSMC gastric cancer grading dataset that has been introduced in the paper: 

![](gastric_tissue_sample.png)

## Download
Google drive:\
Training + Validation + Testing (Resize to 512): [[link]](https://drive.google.com/file/d/1KsLvqNdwAnw_WunVyOqi-K/view?usp=sharing)

## Brief description
The tissue images and annotations are provided by Kangbuk Samsung Hospital, Seoul, South Korea. \
Two pathologists have delineated the annotation: Kim, Kyungeun, and Song, Boram.\
Herein, we obtained the benign (BN) and three cancer ROIs, including tubular well-differentiated adenocarcinoma (TW), tubular moderately-differentiated adenocarcinoma (TM), and tubular poorly-differentiated adenocarcinoma (TP) tumor. 

- The train+valid+test sets contain 98 whole slide images (WSIs) from 98 patients that were collected between 2016 and 2020 from Kangbuk Samsung Hospital (IRB No. 2021-04-035) and scanned at 40x magnification using an Aperio digital slide scanner (Leica Biosystems). 

- The patch images are generated at 40x of size (~270 &micro;m x 0.270 &micro;m), then resize to 512x512 pixels (20x).
For more detail, please refer to the paper above.


## Dataset detail
| **Status** | **Training** | **Validation** | **Testing** |
|------------|--------------|----------------|-------------|
| Benign     | 20,883       | 8,398          | 7,955       |
| TW         | 14,251       | 2,239          | 1,795       |
| TM         | 20,815       | 2,370          | 2,458       |
| TP         | 27,689       | 2,374          | 3,579       |



## Training + Validation + Testing  Structure

KBSMC_gastric_cancer_grading_512 \
├── WSIs \
│ ├── WSIs_001 \
│ │ ├── patch_1152_6592_class_2.jpg \
│ │ ├── patch_1344_6720_class_2.jpg \
│ │ ├── patch_1344_6848_class_2.jpg \
│ │ ├── ... \
│ ├── WSIs_002 \
│ ├── WSIs_003 \
│ ├── WSIs_004 \
│ ├── ... \
│ └── WSIs_158 \
└── WSIs_Split.csv

### Notes:
Train+Valid+Test 
The class labels are determined by the last digit in the image name (bolded), and there are 5 digits  from 1 to 5:

1:"BN" 2:"BN", 3:"TW", 4:"TM", 5:"TP"

We consolidate 1 and 2. Specifically, when the last digit is 1 or 2, the class is considered "Benign." 

For example, if the image name is "patch_XXXX_YYYY_class_1.jpg," it belongs to the benign class. This categorization aligns with the usage of four datasets in our paper.

Please refer to the data loading code for more details.

## Simple way to load the dataset
Check out the dataset.py

```python
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

```

# Citation
