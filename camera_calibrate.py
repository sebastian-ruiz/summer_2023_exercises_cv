import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm
import pickle

# https://github.com/bvnayak/stereo_calibration/tree/master

class StereoCalibration(object):
    def __init__(self, filepath):

        self.visualise = True

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        self.board_rows = 9
        self.board_columns = 6
        self.square_length = 0.021 # 3.5cm
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.board_rows*self.board_columns, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_rows, 0:self.board_columns].T.reshape(-1, 2)*self.square_length

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + '/RIGHT/*.jpg')
        images_left = glob.glob(cal_path + '/LEFT/*.jpg')
        images_left.sort()
        images_right.sort()

        for i in tqdm(np.arange(len(images_right))):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.board_rows, self.board_columns), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.board_rows, self.board_columns), None)


            if ret_l is True and ret_r is True:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (self.board_rows, self.board_columns),
                                                  corners_l, ret_l)
                
                # cv2.waitKey(500)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (self.board_rows, self.board_columns),
                                                  corners_r, ret_r)
                
                if self.visualise:
                    cv2.imshow("left", img_l)
                    cv2.imshow("right", img_r)

                    if cv2.waitKey(1) == ord('q'):
                        break

            img_shape = gray_l.shape[::-1]

        print("running calibration...")

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', K1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', K2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('K1', K1), ('K2', K2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])
        
        with open(f'{self.cal_path}.pickle', 'wb') as f:
            pickle.dump(camera_model, f)

        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default="recording7_conference_calib", help='String Filepath')
    args = parser.parse_args()
    print("args.filepath", args.filepath)
    cal_data = StereoCalibration(args.filepath)
