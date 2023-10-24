import os
from natsort import os_sorted
import cv2
import rospy
from ros_subscriber import ROSSubscriber
from tqdm import tqdm
import pickle


def _get_img_list(dataset_dir):
    img_pairs = []
    for subdir, dirs, files in tqdm(list(os.walk(os.path.join(dataset_dir, "LEFT")))):
        files = os_sorted(files)
        for file in files:
            filepath = os.path.join(subdir, file)
            subfolder = os.path.basename(subdir)
            gt = None
            if subfolder == "LEFT" or subfolder == "RIGHT":
                left_path = os.path.join(dataset_dir, "LEFT", file)
                right_path = os.path.join(dataset_dir, "RIGHT", file)
            else:
                left_path = os.path.join(dataset_dir, "LEFT", subfolder, file)
                right_path = os.path.join(dataset_dir, "RIGHT", subfolder, file)
                gt = subfolder
            if os.path.isfile(right_path):
                img_pairs.append((left_path, right_path, gt))
            else:
                print("[red]not existing", left_path)

    # load the calibration file if it exists
    calibration = None
    for file in os.listdir(os.path.join(dataset_dir)):
        if file.endswith(".pickle"):
            with open(os.path.join(dataset_dir, file), 'rb') as f:
                calibration = pickle.load(f)
            break

    return img_pairs, calibration

class ImgLoader():
    def __init__(self, use_ros=True, folder=None) -> None:
        self.use_ros = use_ros
        self.folder = folder
        self.calibration = None

        if use_ros:
            self.calibration = 'results/recording7_conference_calib.pickle'
            self.ros_subscriber = ROSSubscriber()
        else:
            # rospy.init_node("client")
            self.img_list, self.calibration = _get_img_list(folder)

    def next(self):
        if self.use_ros:

            rate = rospy.Rate(30)
            while not rospy.is_shutdown():
                img1, img2 = self.ros_subscriber.read()
                yield img1, img2, None

                rate.sleep()
        else:
            for img1_path, img2_path, gt in self.img_list:
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                yield img1, img2, gt
