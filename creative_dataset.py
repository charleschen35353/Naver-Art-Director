from torchvision import datasets
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from skimage import io, transform
import numpy as np
from PIL import Image

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.imgs[index]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CreativeDataset(Dataset):
    """Ad Creatives dataset."""

    def __init__(self, csv_file, root_dir, args, transform=None, mapping=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mapping (list: (int, int, string), optional): Optional parameter for k-fold
                cross validation.
        """
        self.item = pd.read_csv(csv_file, header=None)

        if args.short_data :
            self.item = self.item.ix[:9]
        self.classes = self.item.iloc[:,1].unique()
        self.root_dir = root_dir
        self.transform = transform
        self.map = []
        if mapping is not None:
            k, target, mode = mapping
            for i in range(len(self.item)):
                if (i % k == target and mode == 'valid') or \
                    (i % k != target and mode == 'train'):
                    self.map.append(i)


    def __len__(self):
        return len(self.map) if self.map else len(self.item)

    def __getitem__(self, idx):
        if self.map:
            idx = self.map[idx]
        if self.transform == 'Preprocessed':
            img_name = './train_cache/' + self.item.iloc[idx, 0].split('/')[-1] + '.pt'
            image = torch.load(img_name)
        else:
            img_name = os.path.join(self.root_dir,
                                    self.item.iloc[idx, 0])
            image = default_loader(img_name)    
            if self.transform:
                image = self.transform(image)
        label = self.item.iloc[idx, 1]
        ctr = self.item.iloc[idx, 3]

        sex = 0
        age_area = '-'

        try:
            sex = self.item.iloc[idx, 4]
            age_area = self.item.iloc[idx, 5]
        except Exception as e:
            print("e:", e)

        if sex == 'm':
            sex = 1
        else:
            sex = 0

        if age_area == '-14':
            age_area = 0
        elif age_area == '15-19':
            age_area = 1
        elif age_area == '20-24':
            age_area = 2
        elif age_area == '25-29':
            age_area = 3
        elif age_area == '30-34':
            age_area = 4
        elif age_area == '35-39':
            age_area = 5
        elif age_area == '40-44':
            age_area = 6
        elif age_area == '45-49':
            age_area = 7
        elif age_area == '50-':
            age_area = 8
        else:
            age_area = 9
        age = np.zeros((10))

        if age_area <= 8:
            age[age_area] = 1

        if label == 0:
            ordered_label = np.array([0, 0, 0, 0])
        elif label == 1:
            ordered_label = np.array([1, 0, 0, 0])
        elif label == 2:
            ordered_label = np.array([1, 1, 0, 0])
        elif label == 3:
            ordered_label = np.array([1, 1, 1, 0])
        else :
            ordered_label = np.array([1, 1, 1, 1])


        # print(image.size())


        # print('image:', image, 'label:',label)

        # sample = {'image': image, 'label':label, 'sex':sex, 'age_area':age}

        # print(sample)

        return image, label, ctr * 100, sex, age, self.item.iloc[idx, 2]
class CreativeDataset_contage(Dataset):
    """Ad Creatives dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.item = pd.read_csv(csv_file, header=None)
        self.classes = self.item.iloc[:,1].unique()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.item.iloc[idx, 0])

        image = default_loader(img_name)
        label = self.item.iloc[idx, 1]
        ctr = self.item.iloc[idx, 3]

        sex = 0
        age_area = '-'

        try:
            sex = self.item.iloc[idx, 4]
            age_area = self.item.iloc[idx, 5]
        except Exception as e:
            print("e:", e)

        if sex == 'm':
            sex = 1
        else:
            sex = 0

        if age_area == '-14':
            age_area = 0
        elif age_area == '15-19':
            age_area = 1
        elif age_area == '20-24':
            age_area = 2
        elif age_area == '25-29':
            age_area = 3
        elif age_area == '30-34':
            age_area = 4
        elif age_area == '35-39':
            age_area = 5
        elif age_area == '40-44':
            age_area = 6
        elif age_area == '45-49':
            age_area = 7
        elif age_area == '50-':
            age_area = 8
        else:
            age_area = 9
        age = np.zeros((10))

        if age_area <= 8:
            age[age_area] = 1

        if label == 0:
            ordered_label = np.array([0, 0, 0, 0])
        elif label == 1:
            ordered_label = np.array([1, 0, 0, 0])
        elif label == 2:
            ordered_label = np.array([1, 1, 0, 0])
        elif label == 3:
            ordered_label = np.array([1, 1, 1, 0])
        else :
            ordered_label = np.array([1, 1, 1, 1])

        if self.transform:
            image = self.transform(image)

        return image, label, ctr * 100, sex, age

class CreativeDataset_u(Dataset):
    """Ad Creatives dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.item = pd.read_csv(csv_file, header=None)
        self.classes = self.item.iloc[:,1].unique()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir,
        #                        self.item.iloc[idx, 0])

        #image = default_loader(img_name)
        label = self.item.iloc[idx, 1]
        ctr = self.item.iloc[idx, 3]

        sex = 0
        age_area = '-'

        try:
            sex = self.item.iloc[idx, 4]
            age_area = self.item.iloc[idx, 5]
        except Exception as e:
            print("e:", e)

        print(age_area)
        print(sex)
        if sex == 'm':
            sex = 1
        else:
            sex = 0

        orig_age = age_area
        if age_area == '-14':
            age_area = 0
        elif age_area == '15-19':
            age_area = 1
        elif age_area == '20-24':
            age_area = 2
        elif age_area == '25-29':
            age_area = 3
        elif age_area == '30-34':
            age_area = 4
        elif age_area == '35-39':
            age_area = 5
        elif age_area == '40-44':
            age_area = 6
        elif age_area == '45-49':
            age_area = 7
        elif age_area == '50-':
            age_area = 8
        else:
            age_area = 9
        age = np.zeros((10))

        if age_area <= 8:
            age[age_area] = 1

        if label == 0:
            ordered_label = np.array([0, 0, 0, 0])
        elif label == 1:
            ordered_label = np.array([1, 0, 0, 0])
        elif label == 2:
            ordered_label = np.array([1, 1, 0, 0])
        elif label == 3:
            ordered_label = np.array([1, 1, 1, 0])
        else :
            ordered_label = np.array([1, 1, 1, 1])


        # print(image.size())

        #if self.transform:
        #    image = self.transform(image)

        # print('image:', image, 'label:',label)

        # sample = {'image': image, 'label':label, 'sex':sex, 'age_area':age}

        # print(sample)

        #return image, label, ctr * 100, sex, age
        return  label, ctr * 100, sex, age,age_area,orig_age
