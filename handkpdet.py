# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:00:49 2023

@author: LSJ
"""

import os
import cv2
import mmcv
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib

from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline

from mmengine.registry import init_default_scope

from mmpose_models.utils import inference_topdown, inference_detector_batch, inference_topdown_batch, kp2bbox

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class HandKPDetBatch():
    '''
    detect the key points for hand. input could be one image(ndarray/str) or list of images
    '''
    def __init__(self, **kwargs):
        args = DotDict()
        args.det_cat_id=0
        args.bbox_thr=0.3
        args.nms_thr=0.3
        args.draw_heatmap=False
        args.draw_bbox=True
        args.show_kpt_idx=False
        args.skeleton_style='mmpose'
        args.show=False
        args.kpt_thr=0.1
        args.det_model='mmpose_models/rtmdet_nano_320-8xb32_hand.py'
        args.det_weights='mmpose_models/rtmdet_nano_320-8xb32_hand_epoch_390.pth'
        args.pose2d='mmpose_models/rtmpose-m_8xb256-210e_hand5-256x256.py'
        args.pose2d_weights='mmpose_models/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        args.device='cpu'
        args.min_valid_kps=0
        self.args = args
        self.args.update(kwargs)
        self.init_inferencer()
    
    def init_inferencer(self, **kwargs):
        self.args.update(kwargs)
        # build detector
        print(self.args.det_model)
        init_default_scope('mmdet')
        if self.args.det_model:
            self.detector = init_detector(
                self.args.det_model,
                self.args.det_weights,
                device=self.args.device)
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        else:
            self.detector = None
        
        # build pose estimator
        print(self.args.pose2d)
        init_default_scope('mmpose')
        self.pose_estimator = init_pose_estimator(
            self.args.pose2d,
            self.args.pose2d_weights,
            device=self.args.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=self.args.draw_heatmap))))
        
        # build visualizer
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta)
        
    def process(self,
                args,
                img_list,
                show_interval=0):
        """Visualize predicted keypoints (and heatmaps) of one image."""
        if not isinstance(img_list, list):
            img_list = [img_list]
        det_result_list = inference_detector_batch(self.detector, img_list)
        bboxes_list = []
        for det_result in det_result_list:
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                           pred_instance.scores > args.bbox_thr)]
            bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
            bboxes_list.append(bboxes)
        
        pose_results_list = inference_topdown_batch(self.pose_estimator, img_list, bboxes_list)
        result_list = []
        vis_list = []
        for i, img in enumerate(img_list):
            pose_results = pose_results_list[i]
            pose_results = [pose for pose in pose_results if np.sum(pose.pred_instances.get('keypoint_scores')>args.kpt_thr) > args.min_valid_kps]
            if len(pose_results) == 0:
                data_samples = PoseDataSample()
            else:
                data_samples = merge_data_samples(pose_results)
            # show the results
            if isinstance(img, str):
                img = mmcv.imread(img, channel_order='rgb')
            elif isinstance(img, np.ndarray):
                img = mmcv.bgr2rgb(img)
        
            vis = self.visualizer.add_datasample(
                'result',
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=args.draw_heatmap,
                draw_bbox=args.draw_bbox,
                show_kpt_idx=args.show_kpt_idx,
                skeleton_style=args.skeleton_style,
                show=args.show,
                wait_time=show_interval,
                kpt_thr=args.kpt_thr)
            result_list.append(data_samples.get('pred_instances'))
            vis_list.append(vis)
        return result_list, vis_list
    
    def __call__(self, img, **kwargs):
        """Call the inferencer.
        Args:
            img (np.ndarray | str): The loaded image or image file to inference
        """
        self.args.update(kwargs)
        result, vis = self.process(self.args, img)
        return result, vis
    
    def save_img(self, img, save_name, save_path='./results', input_color_order='rgb'):
        """img (ndarray) – Image array to be written.
        """
        if input_color_order=='rgb': img = img[...,::-1]
        out_file = os.path.join(save_path, save_name)
        mmcv.imwrite(img, out_file)
    
    def show_img(self, img, input_color_order='rgb'):
        """img (ndarray) – Image array to be written.
        """
        if input_color_order=='bgr': img = img[...,::-1]
        plt.figure(dpi=300)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
            

    def show_img_kp(self, img, kp, input_color_order='rgb'):
        """img (ndarray) – Image array to be written.
           kp (ndarray) – shape=21*2 keypoints for one hand.
        """
        if input_color_order=='bgr': img = img[...,::-1]
        plt.figure(dpi=300)
        plt.clf()
        plt.imshow(img)
        edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
        for ie, e in enumerate(edges):
            rgb = matplotlib.colors.hsv_to_rgb([ie/float(len(edges)),1.0,1.0])
            plt.plot(kp[e,0],kp[e,1],color=rgb)
        plt.axis('off')
        plt.show()

def filter_onehand(keypoints_list, keypoint_scores_list):
    # TODO: exercise 1 -->
    onehand_keypoints_list = []
    onehand_keypoint_scores_list = []
    for kp, kp_sc in zip(keypoints_list, keypoint_scores_list):
        pass
    # TODO: <-- exercise 1
    return onehand_keypoints_list, onehand_keypoint_scores_list


if __name__ == '__main__':
    
    handkpdet = HandKPDetBatch(device='cpu') #detect the key points for hand. input could be one image(ndarray/str) or list of images
    
    # inference
    img_path = 'models/demo.jpg'  # replace this with your own image path
    img = cv2.imread(img_path) # bgr order
    
    img_list = [img, img]
    result_list, vis_list = handkpdet(img_list, show=False, draw_bbox=True, show_kpt_idx=True, kpt_thr=0.1)
    keypoints_list = [result.get('keypoints') if result else None for result in result_list]
    keypoint_scores_list = [result.get('keypoint_scores') if result else None for result in result_list]
    onehand_keypoints_list, onehand_keypoint_scores_list = filter_onehand(keypoints_list, keypoint_scores_list)
    
    handkpdet.show_img(vis_list[0])
    handkpdet.show_img_kp(img_list[0], onehand_keypoints_list[0] , input_color_order = 'bgr')

        
