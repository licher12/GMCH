import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np
import scipy.io as scio


def loader(path, dataset):
    if dataset == 'mirflickr':
        data_file = scio.loadmat(path + "FLICKR-25K.mat")
        images = data_file['images'][:]
        images = (images - images.mean()) / images.std()
        tags = data_file['YAll'][:]
        labels = data_file['LAll'][:]

    elif dataset == 'NUS-WIDE-TC21':
        data_file = scio.loadmat(path + "NUS-WIDE-TC21")
        images = data_file['image'][:]
        images = (images - images.mean()) / images.std()
        tags = data_file['text'][:]
        labels = data_file['label'][:]


    return images, tags, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, db_size, training_size, query_size, images, tags, labels, test=None):
        self.test = test
        all_index = np.arange(tags.shape[0])
        if dataset == 'mirflickr':
            query_index = all_index[db_size:]
            training_index = all_index[:training_size]
            db_index = all_index[:db_size]
        else:
            query_index = all_index[:query_size]
            training_index = all_index[query_size: query_size + training_size]
            db_index = all_index[query_size:]

        if test is None:
            train_images = images[training_index]
            train_tags = tags[training_index]
            train_labels = labels[training_index]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[query_index]
            self.db_labels = labels[db_index]
            if test == 'image.query':
                self.images = images[query_index]
            elif test == 'image.db':
                self.images = images[db_index]
            elif test == 'text.query':
                self.tags = tags[query_index]
            elif test == 'text.db':
                self.tags = tags[db_index]

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                torch.from_numpy(self.images[index].astype('float32')),
                torch.from_numpy(self.tags[index].astype('float32')),
                torch.from_numpy(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return torch.from_numpy(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )



