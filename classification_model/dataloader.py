# Imports
import torch
from torch.utils.data import Dataset
import numpy as np
import jsonpickle
from sklearn.model_selection import StratifiedKFold


# Constants
CHECKER_SIZE = 0.035 #one mark is 3.5x3.5cm'set


# Label Readout
def get_label_set(data_file):
    with open(data_file, 'r') as json_file:
        pose_dataset = jsonpickle.decode(json_file.read(), keys=True)
    label_names = sorted(list(set([pose_dataset[x].subfolder for x in range(len(pose_dataset))])))
    return label_names

# Split Generation
def get_data_split(data_file, case='3D', num_folds=5):
    with open(data_file, 'r') as json_file:
        pose_dataset = jsonpickle.decode(json_file.read(), keys=True)
    
    if case == '2D':
        data = np.array([pose_dataset[x].points_2d_left for x in range(len(pose_dataset))])
    else:
        data = np.array([pose_dataset[x].points_3d for x in range(len(pose_dataset))])

    label = np.array([pose_dataset[x].subfolder for x in range(len(pose_dataset))])



    skf = StratifiedKFold(n_splits=num_folds)   # for classification
    for train_index, test_index in skf.split(data, label):
        train_set = (data[train_index], label[train_index])
        test_set = (data[test_index], label[test_index])

        yield train_set, test_set



# 2D Dataloader Definition
def _rotationMatrix2D(x):
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


class CustomDataset2D(Dataset):
    def __init__(self, train_data, labels_set, transform=None, augment=True):
        self.transform = transform
        self.augment = augment
        self.points, self.labels = train_data
        self.labels_set = labels_set


    def __len__(self):
        return len(self.points)
    
    
    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]

        points = torch.tensor(points).to(dtype=torch.float32)
        # points /= torch.tensor((1024, 768))
        points = (points / torch.tensor([512,384])) - 1 #(points - torch.tensor([751.8676, 296.0833])) / torch.tensor([90.1121, 65.7381])
        if self.augment:
            # jitter
            rj = torch.rand(21,2)*0.1-0.05 #-0.05,0.05
            points = points + rj
            # rescale
            rs = torch.rand(2)+0.5 #0.5,1.5
            points = points * rs
            # rotate
            center = points.mean(axis=0)
            rr = torch.rand(1)*2-1 #-1,1
            rr = np.deg2rad(rr * 20).numpy().squeeze() #-20,20
            points = (points-center) @ _rotationMatrix2D(rr).T + center
            points = points.squeeze(0)
            # translate
            rt = torch.rand(2)*2-1 #-1,1
            points = points + rt

        points = points.flatten()

        label = self.labels_set.index(label)
            
        return points, label
    

# 3D Dataloader Definition
class CustomDataset3D(Dataset):
    def __init__(self, train_data, labels_set, transform=None, augment=True):
        self.transform = transform
        self.augment = augment
        self.points, self.labels = train_data
        self.labels_set = labels_set

        self.std = torch.tensor([[0.1196, 0.0736, 0.0823]], dtype=torch.float32)
        self.mean = torch.tensor([[ 0.3239, -0.1417,  1.0394]], dtype=torch.float32)


    def __len__(self):
        return len(self.points)


    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]

        points = torch.tensor(points).to(dtype=torch.float32)
        points = points * CHECKER_SIZE
        points = (points - self.mean) / self.std
        if self.augment:
            # jitter
            rj = torch.rand(21,3)*0.01-0.005 #-0.05,0.05
            points = points + rj
            # rescale
            rs = torch.rand(3)+0.5 #0.5,1.5
            points = points * rs
            # translate
            rt = torch.rand(3)*2-1 #-1,1
            points = points + rt

        points = points.flatten()

        label = self.labels_set.index(label)
            
        return points, label
