from hand_pose3d import Pose3d
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import cv2
import io
import tqdm
import multiprocessing

class FIRfilter:
    def __init__(self, ntaps, cutoff, npoints, nd) -> None:
        # TODO: exercise 6 -->
        self.b = None
        # TODO: <-- exercise 6

        self.buffer = np.zeros(shape=(npoints,nd,ntaps))
        self.npoints = npoints
        self.nd = nd

    def filter(self, data):
        # TODO: exercise 6 -->
        z = None
        return z
        # TODO: <-- exercise 6

def gen_save_points():
    pose3d = Pose3d()

    # inference for folder
    from pathlib import Path
    img_folder_path = 'recording4_hand_gestures'   # replace this with your own image path
    left_imgpath_list = sorted(Path(img_folder_path, 'LEFT').glob('*.jpg'), key=lambda x:int(x.stem[6:]))
    right_imgpath_list = sorted(Path(img_folder_path, 'RIGHT').glob('*.jpg'), key=lambda x:int(x.stem[6:]))
    
    # obtain traj_list then animate it
    traj_list = []
    for lp, rp in zip(left_imgpath_list, right_imgpath_list):
        print(lp.stem)
        points_3d, _, _, _, _ = pose3d.stereo_paths_to_3d(str(lp), str(rp), visualise=False)
        if points_3d is not None:
            traj_list.append(points_3d)

    traj_list = np.array(traj_list)
    np.save("recording4_hand_gestures", traj_list)


def plot_3d(p3ds, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_range = p3ds.min(axis=0)
    max_range = p3ds.max(axis=0)
    ax.set_xlim3d(min_range[0], max_range[0])
    ax.set_ylim3d(min_range[1], max_range[1])
    ax.set_zlim3d(min_range[2], max_range[2])
    # xlim=[0, 20]
    # ylim=[-10, 0]
    # zlim=[20, 40]
    # ax.set_xlim3d(xlim)
    # ax.set_ylim3d(ylim)
    # ax.set_zlim3d(zlim)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    connections = [[0,1], [1,2], [2,3], [3,4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for _c in connections:
        ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
    
    fig.savefig("recording4/" + str(i)+".png")
    plt.close(fig)

def animate(traj_list):

    # p = multiprocessing.Pool(8)
    # for i in range(len(traj_list)):
    #     p.apply_async(plot_3d, args=(traj_list[i], i))
    # p.close()
    # p.join()
    
    imgs = []
    for i in range(len(traj_list)):
        imgs.append(cv2.imread("recording4/" + str(i)+".png"))
    fourcc = cv2.VideoWriter.fourcc('M','J','P','G')
    video_size = (imgs[0].shape[1], imgs[0].shape[0])
    video_writer = cv2.VideoWriter("recording4.avi", fourcc, 30.0, video_size, True)
    for i in range(0, len(traj_list)):
        video_writer.write(imgs[i])

if __name__ == "__main__":

    # gen_save_points()

    traj_list = np.load('recording4_hand_gestures.npy')
    # filter = FIRfilter(15, 0.1, 21, 3)
    # filtered_traj = []
    # for each_frame in tqdm.tqdm(traj_list):
    #     filtered_frame = filter.filter(each_frame)
    #     filtered_traj.append(filtered_frame)
    
    # filtered_traj = np.array(filtered_traj)
    filtered_traj = traj_list
    animate(filtered_traj)

