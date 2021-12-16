import os

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageListDataset(Dataset):
    def __init__(self, path_label_list, img_root=None,
                 transform=None,
                 target_transform=None, label_exist=True,
                 loader=pil_loader):
        self.img_root = img_root
        self.data = path_label_list
        self.label_exist = label_exist
        if self.label_exist == False:
            self.data = [[item] for item in path_label_list]

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, i):
        '''
        if label exists, get (img,label_idx) pair of i-th data point
         if label does not exit, just return image tensor of i-th data point
        img is already preprocessed
        label_idx start from 0 incrementally so can be used for cnn input directly
        '''
        if self.label_exist:
            return self.get_img(i), self.get_label_idx(i)
        else:
            return self.get_img(i)

    def get_img_path(self, i):
        '''
        get img_path of i-th data point
        '''
        img_path = self.data[i][0]
        if self.img_root is not None:
            img_path = os.path.join(self.img_root, img_path)
        return img_path

    def get_img(self, i):
        '''
        get img array of i-th data point
        self.transform is applied if exists
        '''
        img = self.loader(self.get_img_path(i))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_label(self, i):
        '''
        get label of i-th data point as it is. 
        '''
        assert self.label_exist
        return self.data[i][1]

    def get_label_idx(self, i):
        '''
        get label idx, which start from 0 incrementally
        self.target_transform is applied if exists
        '''
        label = self.get_label(i)
        if self.target_transform is not None:
            if isinstance(self.target_transform, dict):
                indice_label = self.target_transform[label]
            else:
                indice_label = self.target_transform(label)
        else:
            indice_label = label
        return indice_label

    def __len__(self):
        return len(self.data)
