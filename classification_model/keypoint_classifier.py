import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import KeyPointClassifier as KeyPointClassifierModel


# TODO: exercise 3b -->
NUM_CLASSES =
WEIGHT_2D =
WEIGHT_3D =
# TODO: <-- exercise 3b



class KeyPointClassifier:
    def __init__(self, use_2d=True) -> None:
        self.sm = nn.Softmax()

        self.use_2d = use_2d

        if use_2d:
            input_size = 42
            model_path = WEIGHT_2D
        else:
            input_size = 63
            model_path = WEIGHT_3D


        self.model = KeyPointClassifierModel(input_size, NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path))


        self.model.eval()

        self.labels = ['0_fist', '10_pointing', '11_pointing2', '12_pinch', '13_thumbs_up', '14_c', '15_flat', '16_perfect', '17_startrek', '1_one', '2_two', '3_three', '4_four', '5_five', '6_metal', '7_fingers_crossed', '8_thumbs_down', '9_open_hand']


    def infer(self, points):

        points = torch.tensor(points).to(dtype=torch.float32)
        
        if self.use_2d:
            points = (points / torch.tensor([512,384])) - 1
        else:
            std = torch.tensor([[0.1196, 0.0736, 0.0823]], dtype=torch.float32)
            mean = torch.tensor([[ 0.3239, -0.1417,  1.0394]], dtype=torch.float32)
            points = (points - mean) / std

        points = points.reshape(1, self.input_size)

        with torch.no_grad():
            output = self.sm(self.model(points)).squeeze()

        pred = torch.argmax(output)

        print("output", output.shape, output)

        pred_label = self.labels[pred]
        conf = output[pred].numpy()

        return pred_label, conf
    