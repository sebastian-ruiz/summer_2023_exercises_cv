from timeit import default_timer as timer
from hand_pose3d import Pose3d
import numpy as np
from rich import print
from timeit import default_timer as timer
from helpers import visualise
from classification_model import KeyPointClassifier
from img_loader import ImgLoader


if __name__ == '__main__':
    time1 = timer()

    use_2d = False
    keypoint_classifier = KeyPointClassifier(use_2d)

    dataset_dir = "recording4_hand_gestures"
    img_loader = ImgLoader(use_ros=False, folder=dataset_dir)
    pose3d = Pose3d(img_loader.calibration)
    
    for img_left, img_right, _ in img_loader.next():

        if img_left is not None and img_right is not None:

            time0 = timer()
            points_3d, uvs1, uvs2, vis1, vis2 = pose3d.stereo_paths_to_3d(img_left, img_right)
            time1 = timer()

            text = ""
    
            if points_3d is not None:
                if use_2d:
                    pred_label, conf = keypoint_classifier.infer(uvs1)
                else:
                    pred_label, conf = keypoint_classifier.infer(points_3d)

                time2 = timer()

                print(f"pose time: {time1 - time0}")
                print(f"network prediction time: {time2 - time1}")

                text = f"{pred_label}, {np.round(conf, 3)}"
            
            visualise(vis1, vis2, text, wait=1)
