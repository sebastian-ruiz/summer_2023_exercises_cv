# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:56:33 2023

@author: LSJ
"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation



class PlotTraj():
    def __init__(self):
        self.traj_list=[]
        
    def reset(self):
        self.traj_list=[]
    
    def update(self, num, traj, line):
        # line, = ax.plot(traj[0, 0:1], traj[1, 0:1], traj[2, 0:1], '-', )
        line.set_data(traj[:2, :num])
        line.set_3d_properties(traj[2, :num])
        return line,
    
    def animate(self, traj_list, title='PlotTraj', xlim=[0, 20], ylim=[-10, 0], zlim=[20, 40]):
        '''
        traj_list: list for the trajectory points (1x3 nd.array)
        return matplotlib.animation.Animation, Animation.save('path.gif')
        '''
        self.traj_list = traj_list
        traj = np.concatenate(traj_list,0).T
        N = len(traj_list)
        fig = plt.figure()
        fig.suptitle(f'{title}')
        ax = fig.add_subplot(projection='3d')
        line, = ax.plot(traj[0, 0:1], traj[1, 0:1], traj[2, 0:1], '-', )
        # Setting the axes properties
        ax.set_xlim3d(xlim)
        ax.set_xlabel('X')
        ax.set_ylim3d(ylim)
        ax.set_ylabel('Y')
        ax.set_zlim3d(zlim)
        ax.set_zlabel('Z')
        ani = animation.FuncAnimation(fig, self.update, N, interval=50, fargs=(traj,line), blit=True, repeat=True)
        plt.show()
        return ani

    def plot(self, traj_list, title='PlotTraj', xlim=[0, 20], ylim=[-10, 0], zlim=[20, 40]):
        """
        3d plot
        """
        self.traj_list = traj_list
        traj = np.concatenate(traj_list,0).T
        fig = plt.figure()
        fig.suptitle(f'{title}')
        ax = fig.add_subplot(projection='3d')
        line, = ax.plot(traj[0,...], traj[1,...], traj[2,...], '-', )
        # Setting the axes properties
        ax.set_xlim3d(xlim)
        ax.set_xlabel('X')
        ax.set_ylim3d(ylim)
        ax.set_ylabel('Y')
        ax.set_zlim3d(zlim)
        ax.set_zlabel('Z')
        plt.show()
        return
    
    def plot_online(self, generator, title='PlotTraj', xlim=[0, 20], ylim=[-10, 0], zlim=[20, 40]):
        """
        3d plot
        """
        fig = plt.figure()
        plt.ion()
        if_plot = True
        traj_list = []
        while if_plot:
            try:
                traj_list = next(generator)
            except:
                print('the ploting end for the StopIteration')
                if_plot = False
            if len(traj_list)==0:
                traj = np.ones((3,0))
            else:
                traj = np.concatenate(traj_list,0).T
            self.traj_list = traj_list
            fig.clf()
            fig.suptitle(f'{title}')
            ax = fig.add_subplot(projection='3d')
            line, = ax.plot(traj[0,...], traj[1,...], traj[2,...], '-', )
            # Setting the axes properties
            ax.set_xlim3d(xlim)
            ax.set_xlabel('X')
            ax.set_ylim3d(ylim)
            ax.set_ylabel('Y')
            ax.set_zlim3d(zlim)
            ax.set_zlabel('Z')
            plt.pause(0.1)
        plt.ioff()
        plt.show()
        return

if __name__ == '__main__':
    from hand_pose3d import Pose3d
    pose3d = Pose3d()
    
    pt = PlotTraj()
    
    # inference for folder
    from pathlib import Path
    img_folder_path = 'recording3_writing_seb'   # replace this with your own image path
    left_imgpath_list = sorted(Path(img_folder_path, 'LEFT').glob('*.jpg'), key=lambda x:int(x.stem[6:]))
    right_imgpath_list = sorted(Path(img_folder_path, 'RIGHT').glob('*.jpg'), key=lambda x:int(x.stem[6:]))
    
    # obtain traj_list then animate it
    traj_list = []
    for lp, rp in zip(left_imgpath_list, right_imgpath_list):
        print(lp.stem)
        points_3d, _, _, _, _ = pose3d.stereo_paths_to_3d(str(lp), str(rp))
        if points_3d is not None:
            traj_list.append(points_3d.mean(axis=0,keepdims=1))
        
    ani = pt.animate(traj_list)
    ani.save('results/witing_8.gif')
    
    # or plot traj
    pt.plot(traj_list)
    
    
    # or plot traj online
    def update_traj(left_imgpath_list, right_imgpath_list):
        traj_list = []
        path_gen = zip(left_imgpath_list, right_imgpath_list)
        while True:
            lp, rp = next(path_gen)
            print(lp)
            points_3d, _, _, _, _ = pose3d.stereo_paths_to_3d(str(lp), str(rp))
            if points_3d is not None:
                traj_list.append(points_3d.mean(axis=0,keepdims=1))
            yield traj_list
    gen = update_traj(left_imgpath_list, right_imgpath_list)
    # need a generator that could update the traj_list
    # print(next(gen))
    pt.plot_online(gen)
