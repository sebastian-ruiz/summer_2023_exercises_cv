import os
import sys
import cv2
from timeit import default_timer as timer
from hand_pose3d import Pose3d
import numpy as np
from rich import print
from timeit import default_timer as timer
from helpers import visualise
from matthias_dl_classification.keypoint_classifier import KeyPointClassifier
from img_loader import ImgLoader
from traj_filter import FIRfilter
import rospy
import sensor_msgs.msg
import geometry_msgs.msg

def convert_to_point_cloud_ros(points3D, pose):
    pcloud = sensor_msgs.msg.PointCloud()
    pcloud.header.frame_id = pose

    for point in points3D:
        point_ros = geometry_msgs.msg.Point()
        point_ros.x = point[0]
        point_ros.y = point[1]
        point_ros.z = point[2]
        pcloud.points.append(point_ros)

    return pcloud

if __name__ == '__main__':
    time1 = timer()

    use_2d = False

    keypoint_classifier = KeyPointClassifier(use_2d)
    filter = FIRfilter(15, 0.1, 21, 3)
    pub = rospy.Publisher('/human_finger_points', sensor_msgs.msg.PointCloud, queue_size=10)

    dataset_dir = "recording5_fist_pointing"

    img_loader = ImgLoader(use_ros=False, folder=dataset_dir)
    pose3d = Pose3d(img_loader.calibration)
    
    for img_left, img_right, _ in img_loader.next():

        if img_left is not None and img_right is not None:

            time0 = timer()
            points_3d, uvs1, uvs2, vis1, vis2 = pose3d.stereo_paths_to_3d(img_left, img_right)
            time1 = timer()

            text = ""
    
            if points_3d is not None:
                points_3d_filtered = filter.filter(points_3d)
                
                if use_2d:
                    pred_label, conf = keypoint_classifier.infer(uvs1)
                else:
                    pred_label, conf = keypoint_classifier.infer(points_3d_filtered)

                time2 = timer()
                
                # publish
                pcloud = convert_to_point_cloud_ros(points_3d_filtered, pred_label)
                pub.publish(pcloud)

                print(f"pose time: {time1 - time0}")
                print(f"network prediction time: {time2 - time1}")

                text = f"{pred_label}, {np.round(conf, 3)}"
            
            visualise(vis1, vis2, text, wait=1)
