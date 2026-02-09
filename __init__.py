# scene/__init__.py

import os
import random
import json
import pytorch3d
import pytorch3d.transforms
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from romatch import roma_outdoor

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.dataset_readers_raw import readKittiRawSceneInfo

from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.calibrate_gaussian_model import CalibrateGaussianListModel
from sensors.sensor_trajectories import SensorTrajectories
# [필수] 행렬 재계산을 위한 함수
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Scene:
    gaussians : CalibrateGaussianListModel

    def __init__(self, args : ModelParams, gaussians : CalibrateGaussianListModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], voxel_resolutions = [0.3], sensor_trajectories=None):
        self.model_path = args.model_path
        self.weight_path = args.flow_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.resolution_scales = resolution_scales
        self.voxel_resolutions = voxel_resolutions
        self.current_level = 0

        # [데이터셋 감지]
        is_kitti_360 = os.path.exists(os.path.join(args.source_path, "data_3d_raw"))
        is_kitti_raw = False
        if os.path.exists(os.path.join(args.source_path, "oxts")): is_kitti_raw = True
        elif os.path.isdir(args.source_path):
            try:
                if any("drive" in d and "sync" in d for d in os.listdir(args.source_path)): is_kitti_raw = True
            except: pass
        if not is_kitti_raw and ("drive" in args.source_path or "KITTI_RAW" in args.source_path): is_kitti_raw = True

        # [데이터 로딩]
        if is_kitti_360:
            dataset_type = 'KITTI360'
            print("Detected KITTI-360 dataset")
            scene_info_list, all_lidar_points = sceneLoadTypeCallbacks["KITTI360"](
                root_path=args.source_path, root_image_path=args.source_path, 
                sequence=str(args.data_seq if args.data_seq is not None else 0),
                cam_id=args.cam_id, start_index=args.start_index, segment_length=30
            )
        elif is_kitti_raw:
            dataset_type = 'KITTI_RAW'
            print(f"[Auto] KITTI Raw Loader: {args.source_path}")
            scene_info_list, all_lidar_points = readKittiRawSceneInfo(
                path=args.source_path, images=args.images, eval=args.eval,
                cam_id=getattr(args, 'cam_id', '02'), start_index=getattr(args, 'start_index', 0),
                segment_length=getattr(args, 'segment_length', 50)
            )
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info_list = scene_info_list
        self.all_lidar_points = all_lidar_points

        if not self.loaded_iter:
            with open(scene_info_list.ply_path, 'rb') as src, open(os.path.join(self.model_path, "input.ply") , 'wb') as dst:
                dst.write(src.read())
            json_cams = []
            camlist = []
            if scene_info_list.test_cameras: camlist.extend(scene_info_list.test_cameras)
            if scene_info_list.train_cameras: camlist.extend(scene_info_list.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info_list.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info_list.train_cameras, resolution_scale, args, scene_info_list.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info_list.test_cameras, resolution_scale, args, scene_info_list.is_nerf_synthetic, True)
            
            # [0,0,0 데이터 복구 로직 - Znear 수정 없이 순정 복구]
            if dataset_type == 'KITTI_RAW':
                print("[Safety Check] Verifying Camera Poses...")
                fixed_count = 0
                for i, cam in enumerate(self.train_cameras[resolution_scale]):
                    original_T = scene_info_list.train_cameras[i].T
                    current_T = cam.T
                    
                    if np.allclose(current_T, 0) and not np.allclose(original_T, 0):
                        cam.R = scene_info_list.train_cameras[i].R.copy()
                        cam.T = original_T.copy()
                        
                        # 행렬 재계산 (기존 znear/zfar 값 유지)
                        trans = np.array([0.0, 0.0, 0.0])
                        scale = 1.0
                        cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, trans, scale)).transpose(0, 1).cuda()
                        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0, 1).cuda()
                        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
                        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                        fixed_count += 1
                
                if fixed_count > 0:
                    print(f" -> [RECOVERED] {fixed_count} cameras restored from 0,0,0 state.")

        self.generate_optical_flow()

        initialized_pose_cl = None
        if dataset_type in ['KITTI360', 'KITTI', 'KITTI_RAW']:
            initial_rotation_mat = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            initialized_pose_cl = np.eye(4)
            initialized_pose_cl[:3,:3] = initial_rotation_mat

        self.sensor_trajectory = SensorTrajectories(
            lidar_rotations=scene_info_list.lidar_rotations,
            lidar_translations=scene_info_list.lidar_translations,
            lidar_timestamps=[0. for i in range(len(scene_info_list.lidar_rotations))],
            pose_cl=scene_info_list.pose_cl,
            initialized_pose_cl=initialized_pose_cl
        )

        for resolution_scale in resolution_scales:
            for cam in self.train_cameras[resolution_scale]: cam.generate_gradient_and_pixel_masks()
            for cam in self.test_cameras[resolution_scale]: cam.generate_gradient_and_pixel_masks()

        rotation_wl_list, translations_wl_list = self.sensor_trajectory.get_all_lidar_poses()
        projection_matrix = self.train_cameras[resolution_scales[self.current_level]][0].projection_matrix

        self.gaussians.create_from_pcd_list(
            pcd_list=scene_info_list.point_cloud_list,
            spatial_lr_scale=self.cameras_extent,
            rotations_wl_list=rotation_wl_list,
            translations_wl_list=translations_wl_list,
            projection_matrix=projection_matrix,
            resolution=self.voxel_resolutions[self.current_level]
        )

    def generate_optical_flow(self):
        resolution_scale = self.resolution_scales[-1]
        train_cameras = self.train_cameras[resolution_scale]
        if not os.path.exists(os.path.join(self.weight_path, 'roma_outdoor.pth')): return
        
        dino_weights = torch.load(os.path.join(self.weight_path, 'dinov2_vitl14_pretrain.pth'))
        weights = torch.load(os.path.join(self.weight_path, 'roma_outdoor.pth'))
        H, W = train_cameras[0].image_height, train_cameras[0].image_width
        flow_model = roma_outdoor(device='cuda', weights=weights, dinov2_weights=dino_weights, coarse_res=560, upsample_res=(H, W))

        def generate_optical_flow_one_group(flow_model, cameras, window_size = 2):
            cam_num = len(cameras)
            images = [(camera.original_image.detach().cpu().permute(1,2,0).numpy()*255.0).astype(np.uint8) for camera in cameras]
            all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs = {}, {}, {}, {}
            for w in range(1, window_size+1):
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                flow_refs, flow_matchs, certainty_refs, certainty_matchs = [], [], [], []
                for i in range(cam_num-w):
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    warp, certainty = flow_model.match_image(images[i], images[i+w], device='cuda')
                    flow_refs.append(warp[:, :W, 2:].detach().cpu())
                    flow_matchs.append(warp[:, W:, :2].detach().cpu())
                    certainty_refs.append(certainty[:, :W].detach().cpu())
                    certainty_matchs.append(certainty[:, W:].detach().cpu())
                all_flow_refs[w] = flow_refs; all_flow_matchs[w] = flow_matchs
                all_certainty_refs[w] = certainty_refs; all_certainty_matchs[w] = certainty_matchs
            return all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs

        window_size = 1
        all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs = generate_optical_flow_one_group(flow_model, train_cameras, window_size=window_size)
        del flow_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for resolution_scale in list(self.train_cameras.keys()):
            cameras = self.train_cameras[resolution_scale]
            level_width, level_height = cameras[0].image_width, cameras[0].image_height
            for w in range(1, window_size+1):
                # 리스트에서 값을 꺼냄
                flow_refs, flow_matchs, certainty_refs, certainty_matchs = all_flow_refs[w], all_flow_matchs[w], all_certainty_refs[w], all_certainty_matchs[w]
                
                # CUDA 텐서로 변환된 리스트 생성
                flow_ref_level = [item.to(device='cuda') for item in flow_refs]
                flow_match_level = [item.to(device='cuda') for item in flow_matchs]
                certainty_ref_level = [item.to(device='cuda') for item in certainty_refs]
                certainty_match_level = [item.to(device='cuda') for item in certainty_matchs]

                for i in range(len(flow_refs)):
                    # [수정완료] 여기서 certainty_refs(리스트)가 아니라 certainty_ref_level[i](텐서)를 써야 함
                    flow_ref_level[i] = F.interpolate(flow_ref_level[i][None, :, :, :].permute(0,3,1,2), size=(level_height, level_width), mode='nearest').squeeze(0).permute(1,2,0)
                    flow_match_level[i] = F.interpolate(flow_match_level[i][None, :, :, :].permute(0,3,1,2), size=(level_height, level_width), mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_ref_level[i] = F.interpolate(certainty_ref_level[i][None, None, :, :], size=(level_height, level_width), mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_match_level[i] = F.interpolate(certainty_match_level[i][None, None, :, :], size=(level_height, level_width), mode='nearest').squeeze(0).permute(1,2,0)

                cam_num = len(cameras)
                for i in range(cam_num-w):
                    cameras[i].forward_flows.append(flow_ref_level[i])
                    cameras[i].forward_flow_uids.append(i+w)
                    cameras[i].forward_certainties.append(certainty_ref_level[i])
                    cameras[i+w].backward_flows.append(flow_match_level[i])
                    cameras[i+w].backward_flow_uids.append(i)
                    cameras[i+w].backward_certainties.append(certainty_match_level[i])
            print('Finish generate optical flow')

    def update_level(self, level):
        self.current_level = level
        voxel_resolution = self.voxel_resolutions[self.current_level]
        
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0): return self.train_cameras[scale]
    def getTrainImages(self, scale=1.0):
        train_images = []
        for camera in self.train_cameras[scale]:
            image = camera.original_image.detach().cpu().permute(1, 2, 0).numpy()
            image *= 255
            train_images.append(image.astype(np.uint8))
        return train_images
    def getTestCameras(self, scale=1.0): return self.test_cameras[scale]
    def getTrainCameraByIndex(self, index, scale=1.0): return self.train_cameras[scale][index]