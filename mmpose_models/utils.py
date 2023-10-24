# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:21:30 2023

@author: LSJ
"""



from typing import List
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from PIL import Image
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
from typing import  Optional, Union

def kp2bbox(kp, imgshape=None, aug=2):
    """
    Parameters
    ----------
    kp : (np.array) shape: BxNx2

    Returns bbox: xyxy
    -------
    """
    bxmin = kp[:,:,0].min(axis=1, keepdims=True)
    bymin = kp[:,:,1].min(axis=1, keepdims=True)
    bxmax = kp[:,:,0].max(axis=1, keepdims=True)
    bymax = kp[:,:,1].max(axis=1, keepdims=True)
    bboxes = np.concatenate([bxmin,bymin,bxmax,bymax], 1)
    if aug:
        bboxes[:,0] -= (bboxes[:,2]-bboxes[:,0])/2*(aug-1)
        bboxes[:,1] -= (bboxes[:,3]-bboxes[:,1])/2*(aug-1)
        bboxes[:,2] += (bboxes[:,2]-bboxes[:,0])/2*(aug-1)
        bboxes[:,3] += (bboxes[:,3]-bboxes[:,1])/2*(aug-1)
    if imgshape:
        bboxes[:,[0,2]] = bboxes[:,[0,2]].clip(0,imgshape[1])
        bboxes[:,[1,3]] = bboxes[:,[1,3]].clip(0,imgshape[0])
    return bboxes

def inference_topdown(model: nn.Module,
                      img: Union[np.ndarray, str],
                      bboxes: Optional[Union[List, np.ndarray]] = None) -> List[PoseDataSample]:

    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if bboxes is None or len(bboxes) == 0:
        bboxes = np.empty((0,4))
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []
    return results

def inference_topdown_batch(model: nn.Module,
                      img_list,
                      bboxes_list,
                      bbox_format: str = 'xyxy') -> List[PoseDataSample]:
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)


    for i in range(len(bboxes_list)):
        bboxes = bboxes_list[i]
        if bboxes is None or len(bboxes) == 0:
            bboxes_list[i] = np.empty((0,4))
        if isinstance(bboxes, list):
            bboxes_list[i] = np.array(bboxes)

    # construct batch data samples
    data_list = []
    for i in range(len(bboxes_list)):
        bboxes = bboxes_list[i]
        img = img_list[i]
        for bbox in bboxes:
            if isinstance(img, str):
                data_info = dict(img_path=img)
            else:
                data_info = dict(img=img)
            data_info['bbox'] = bbox[None]  # shape (1, 4)
            data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []
    
    results_list = []
    st = 0
    for i in range(len(bboxes_list)):
        bboxes = bboxes_list[i]
        results_list.append(results[st:st+len(bboxes)])
        st += len(bboxes)
            
    return results_list


import copy
from mmcv.ops import RoIPool
# import torch
import torch.nn as nn
# import cv2
# import mmcv
# from mmcv.transforms import Compose
# from mmengine.utils import track_iter_progress
# from mmdet.registry import VISUALIZERS
# from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config, ConfigDict
from typing import Optional, Sequence, Union

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])
        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')
    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)

def inference_detector_batch(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = copy.deepcopy(cfg)
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    data_inputs = {}
    data_inputs['inputs'] = []
    data_inputs['data_samples'] = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_inputs['inputs'] += [data_['inputs']]
        data_inputs['data_samples'] += [data_['data_samples']]

    # forward the model
    with torch.no_grad():
        result_list = model.test_step(data_inputs)
    return result_list
