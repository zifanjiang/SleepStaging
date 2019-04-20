# dataset 

import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split, KFold
from PIL import Image

class cpc(data.Dataset):
    def __init__(self, rdata, split='train', fold=0, transform=None):
        self.transform = transform
        self.split = split
        # xtrain, xtest, ytrain, ytest = train_test_split(rdata['x'], rdata['y'], test_size=0.1, random_state=42)
        x = rdata['x']
        y = rdata['y']
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        kf.get_n_splits(x)
        cnt = 0
        for train_index, test_index in kf.split(x):
            if cnt == fold:
                xtrain, xtest = x[train_index], x[test_index]
                ytrain, ytest = y[train_index], y[test_index]
            cnt += 1
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.05, random_state=42)
        if self.split == 'train':
            self.data = xtrain
            self.labels = ytrain
        elif self.split == 'test':
            self.data = xtest
            self.labels = ytest
        else:
            self.data = xval
            self.labels = yval     

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)