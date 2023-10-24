import cv2
import matplotlib.pyplot as plt
from handkpdet import HandKPDetBatch
import numpy as np
import pickle
from img_loader import ImgLoader
from helpers import visualise


class Pose3d():
    def __init__(self,
                 calibration_file='recording1_calibration.pickle',
                 device="cpu"):
        
        self.device = device

        if isinstance(calibration_file, str):
            with open(calibration_file, 'rb') as f:
                calibration = pickle.load(f)
        else:
            calibration = calibration_file
                

        K1 = calibration['K1']
        K2 = calibration['K2']
        dist1 = calibration['dist1']
        dist2 = calibration['dist2']
        R = calibration['R']
        T = calibration['T']

        # TODO: exercise 2 -->
        
        # TODO: <-- end exercise 2

        # instantiate the inferencer
        self.handkpdet = HandKPDetBatch(device=device)

    def stereo_paths_to_3d(self, left_path, right_path):
        '''
        inputs: could be path(str) or img(nd.array)
        '''
        if isinstance(left_path, str):
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
        else:
            left_img = left_path
            right_img = right_path

        height, width = left_img.shape[:2]
        left_img = left_img[:height - 200]
        right_img = right_img[:height - 200]

        imgs = [left_img, right_img]
        
        result_list, vis_list = self.handkpdet(imgs, show=False, return_vis=True)


        prediction_kp1, prediction_kp2 = [result.get('keypoints') if result else None for result in result_list]
        vis1, vis2 = vis_list
        
        vis1 = vis1[...,::-1]
        vis2 = vis2[...,::-1]

        vis1 = np.require(vis1, requirements=["C_CONTIGUOUS"])
        vis2 = np.require(vis2, requirements=["C_CONTIGUOUS"])
        
        if prediction_kp1 is None or prediction_kp2 is None:
            return None, None, None, vis1, vis2

        # take the first hand only
        uvs1 = np.array(prediction_kp1)[0]
        uvs2 = np.array(prediction_kp2)[0]

        # TODO: exercise 2 -->
        p3ds = None
        # TODO: <-- exercise 2

        return p3ds, uvs1, uvs2, vis1, vis2
    
    def plot_3d(self, p3ds):
        min_range = p3ds.min(axis=0)
        max_range = p3ds.max(axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(min_range[0], max_range[0])
        ax.set_ylim3d(min_range[1], max_range[1])
        ax.set_zlim3d(min_range[2], max_range[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        connections = [[0,1], [1,2], [2,3], [3,4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

        for _c in connections:
            ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
        
        plt.show()


if __name__ == '__main__':

    # left_path, right_path = ['recording4_hand_gestures/LEFT/frame-2644.jpg', 'recording4_hand_gestures/RIGHT/frame-2644.jpg']

    # dataset_dir = "recording1_writing_8"
    dataset_dir = "recording4_hand_gestures"
    img_loader = ImgLoader(use_ros=False, folder=dataset_dir)
    pose3d = Pose3d(img_loader.calibration)
    
    for img_left, img_right, gt in img_loader.next():
        if img_left is not None and img_right is not None:
            points_3d, uvs1, uvs2, vis1, vis2 = pose3d.stereo_paths_to_3d(img_left, img_right)
            if points_3d is not None:
                visualise(vis1, vis2, "", backend="matplotlib")
                pose3d.plot_3d(points_3d)
