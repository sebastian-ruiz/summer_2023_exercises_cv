import mmcv
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer
import os
import cv2
from handkpdet import HandKPDet
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import pickle
import jsonpickle
from tqdm import tqdm
from rich import print
from types import SimpleNamespace
from hand_pose3d import Pose3d
import shutil


if __name__ == '__main__':
    # visualise the results by plotting?
    visualise = False
    save_annotated_imgs = True
    pose3d = Pose3d()

    dataset_stereo_pairs = []
    dataset_dir = "recording4_hand_gestures_subset"
    big_data_dir = "recording4_hand_gestures"

    all_pairs_present = True

    for subdir, dirs, files in tqdm(list(os.walk(os.path.join(dataset_dir, "LEFT")))):
        for file in files:
            filepath = os.path.join(subdir, file)
            subfolder = os.path.basename(subdir)

            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                print("subfolder",subfolder , "file", file)
                left_path = os.path.join(dataset_dir, "LEFT", subfolder, file)
                right_path = os.path.join(dataset_dir, "RIGHT", subfolder, file)
                if os.path.isfile(right_path):
                    dataset_stereo_pairs.append([subfolder, left_path, right_path])
                else:
                    all_pairs_present = False

                    missing_file = os.path.join(dataset_dir, "RIGHT", subfolder, file)
                    copy_from = os.path.join(big_data_dir, "RIGHT", file)
                    print(f"[red]FILE IS MISSING: {missing_file}")
                    print(f"[red]copying from: {copy_from}")

                    shutil.copyfile(copy_from, missing_file)

    if all_pairs_present:
    
        results = []
        for subfolder, left_path, right_path in tqdm(dataset_stereo_pairs):

            p3ds, uvs1, uvs2, vis1, vis2 = pose3d.stereo_paths_to_3d(left_path, right_path)

            # TODO: check that these results are actually good!

            if uvs1 is not None and uvs2 is not None and uvs1.shape == (21, 2) and uvs2.shape == (21, 2):

                result_item = SimpleNamespace()
                result_item.subfolder = subfolder
                result_item.left_path = left_path
                result_item.right_path = right_path
                result_item.points_3d = p3ds
                result_item.points_2d_left = uvs1
                result_item.points_2d_right = uvs2
                
                results.append(result_item)

                if visualise:
                    pose3d.plot_3d(p3ds)

                if save_annotated_imgs:
                    
                    if not os.path.exists(os.path.join("results", os.path.dirname(left_path))):
                        os.makedirs(os.path.join("results", os.path.dirname(left_path)))

                    print("path", os.path.join("results", left_path))
                    img_vis = cv2.hconcat([vis1, vis2])

                    cv2.imwrite(os.path.join("results", left_path), img_vis)

        print("results", len(results))
        # pickle detections
        results_json_str = jsonpickle.encode(results, keys=True, warn=True, indent=2)

        save_path = os.path.join("results", dataset_dir + ".json")
        # write detections to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(results_json_str)

        print(f"[green]saved annotations to: {save_path}")
    
    else:
        print(f"[red]Rerun because some pairs were missing.")