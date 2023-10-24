import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import jsonpickle
from rich import print
from hand_pose3d import Pose3d
from timeit import default_timer as timer
from helpers import visualise
from img_loader import ImgLoader


def calc_error(pts_3d_1, pts_3d_2, visualise=True):

    # TODO: exercise 3a -->

    # TODO: <-- exercise 3a

    return error


class TemplateMatching():
    def __init__(self) -> None:
        # load dataset
        with open("results/recording4_hand_gestures_subset.json", 'r') as json_file:
            self.pose_dataset = jsonpickle.decode(json_file.read(), keys=True)
            print("pose_dataset", len(self.pose_dataset))

    def fit(self, points_3d):
        
        # iterate over dataset. Find pose that fits best
        results = []
        for pose in self.pose_dataset:
            error = calc_error(points_3d, pose.points_3d)
            results.append(error)

        results = np.array(results)
        idx_min = np.argmin(results)
        error = results[idx_min]

        best_pose = self.pose_dataset[idx_min]

        return best_pose.subfolder, error

if __name__ == '__main__':
        # left_path, right_path = ['recording4_hand_gestures/LEFT/frame-2644.jpg', 'recording4_hand_gestures/RIGHT/frame-2644.jpg']

        dataset_dir = "recording4_hand_gestures"
        img_loader = ImgLoader(use_ros=False, folder=dataset_dir)
        pose3d = Pose3d(img_loader.calibration)
        template_matching = TemplateMatching()
        

        for img_left, img_right, gt in img_loader.next():

            if img_left is not None and img_right is not None:
            
                time0 = timer()
                points_3d, uvs1, uvs2, vis1, vis2 = pose3d.stereo_paths_to_3d(img_left, img_right)
                time1 = timer()

                text = ""
                if points_3d is not None:
                
                    name, error = template_matching.fit(points_3d)
                    time2 = timer()

                    print(f"best_pose: {name} error: {error}")

                    print(f"pose time: {time1 - time0}")
                    print(f"template matching time: {time2 - time1}")

                    text = f"{name}, {np.round(error, 3)}, 3d: {points_3d.shape} uvs1: {uvs1.shape}, uvs2: {uvs2.shape}"
                
                visualise(vis1, vis2, text, wait=1)
