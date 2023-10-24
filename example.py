import os
import sys
import cv2
from timeit import default_timer as timer
import numpy as np
from rich import print
from helpers import visualise
from img_loader import ImgLoader


if __name__ == '__main__':
    dataset_dir = "recording4_hand_gestures"
    # dataset_dir = "recording1_writing_8"
    img_loader = ImgLoader(use_ros=True, folder=dataset_dir)
    
    for img_left, img_right, gt in img_loader.next():
        if img_left is not None and img_right is not None:
            print("img_left", img_left)
            visualise(img_left, img_right, "", wait=1)
