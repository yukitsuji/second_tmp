from pathlib import Path

import numpy as np
import torch

import torchplus
from second.core import box_np_ops
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder, second_builder
from second.pytorch.models.voxelnet import VoxelNet
from second.pytorch.train import predict_to_kitti_label, example_convert_to_torch

import abc
import contextlib

from google.protobuf import text_format

from second.data.preprocess import merge_second_batch, prep_pointcloud
from second.protos import pipeline_pb2
import second.data.kitti_common as kitti

class InferenceContext:
    def __init__(self):
        self.config = None
        self.root_path = None
        self.target_assigner = None
        self.voxel_generator = None
        self.anchor_cache = None
        self.built = False

    def get_inference_input_dict(self, info, points):
        assert self.anchor_cache is not None
        assert self.target_assigner is not None
        assert self.voxel_generator is not None
        assert self.config is not None
        assert self.built is True
        kitti.convert_to_kitti_info_version2(info)
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]

        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']

        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second

        input_dict = {
            'points': points,
            "calib": {
                'rect': rect,
                'Trv2c': Trv2c,
                'P2': P2,
            },
            "image": {
                'image_shape': np.array(image_info["image_shape"], dtype=np.int32),
                'image_idx': image_info['image_idx'],
                'image_path': image_info['image_path'],
            },
        }
        out_size_factor = np.prod(model_cfg.rpn.layer_strides)
        if len(model_cfg.rpn.upsample_strides) > 0:
            out_size_factor /= model_cfg.rpn.upsample_strides[-1]
        out_size_factor *= model_cfg.middle_feature_extractor.downsample_factor
        out_size_factor = int(out_size_factor)
        example = prep_pointcloud(
            input_dict=input_dict,
            root_path=str(self.root_path),
            voxel_generator=self.voxel_generator,
            target_assigner=self.target_assigner,
            max_voxels=input_cfg.max_number_of_voxels,
            class_names=self.target_assigner.classes,
            training=False,
            create_targets=False,
            shuffle_points=input_cfg.shuffle_points,
            generate_bev=False,
            without_reflectivity=model_cfg.without_reflectivity,
            num_point_features=model_cfg.num_point_features,
            anchor_area_threshold=input_cfg.anchor_area_threshold,
            anchor_cache=self.anchor_cache,
            out_size_factor=out_size_factor,
            out_dtype=np.float32)
        example["metadata"] = {}
        if "image" in info:
            example["metadata"]["image"] = input_dict["image"]

        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        #############
        # convert example to batched example
        #############
        example = merge_second_batch([example])
        return example

    def get_config(self, path):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        return config

    @abc.abstractclassmethod
    def _build(self):
        raise NotImplementedError()

    def build(self, config_path):
        self.config = self.get_config(config_path)
        ret = self._build()
        self.built = True
        return ret

    @abc.abstractclassmethod
    def _inference(self, example):
        raise NotImplementedError()

    def inference(self, example):
        return self._inference(example)

    @abc.abstractclassmethod
    def _restore(self, ckpt_path):
        raise NotImplementedError()

    def restore(self, ckpt_path):
        return self._restore(ckpt_path)

    @abc.abstractclassmethod
    def _ctx(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def ctx(self):
        yield self._ctx()


class TorchInferenceContext(InferenceContext):
    def __init__(self):
        super().__init__()
        self.net = None
        self.anchor_cache = None

    def _build(self):
        config = self.config
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        train_cfg = config.train_config
        batch_size = 1
        voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        grid_size = voxel_generator.grid_size
        self.voxel_generator = voxel_generator
        vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)

        box_coder = box_coder_builder.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(
            target_assigner_cfg, bv_range, box_coder)
        self.target_assigner = target_assigner
        out_size_factor = model_cfg.rpn.layer_strides[0] / model_cfg.rpn.upsample_strides[0]
        out_size_factor *= model_cfg.middle_feature_extractor.downsample_factor
        out_size_factor = int(out_size_factor)
        self.net = second_builder.build(model_cfg, voxel_generator,
                                          target_assigner)
        self.net.cuda().eval()
        if train_cfg.enable_mixed_precision:
            self.net.half()
            self.net.metrics_to_float()
            self.net.convert_norm_to_float(self.net)
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict,
        }
        self.anchor_cache = anchor_cache

    def _restore(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.suffix == ".tckpt"
        torchplus.train.restore(str(ckpt_path), self.net)

    def _inference(self, example):
        train_cfg = self.config.train_config
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second
        example_torch = example_convert_to_torch(example)
        result_annos = predict_to_kitti_label(
            self.net, example_torch, list(
                self.target_assigner.classes),
            model_cfg.post_center_limit_range, model_cfg.lidar_input)
        return result_annos

    def _ctx(self):
        return None
