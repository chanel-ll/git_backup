#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
import random
import json
import pytorch3d
import pytorch3d.transforms
import cv2

from utils.system_utils import searchForMaxIteration
#from scene.dataset_readers import sceneLoadTypeCallbacks
# KITTI Raw 리더 함수 임포트
from scene.dataset_readers import readKittiRawSceneInfo

from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.calibrate_gaussian_model import CalibrateGaussianListModel
from sensors.sensor_trajectories import SensorTrajectories
import numpy as np
from romatch import roma_outdoor
import torch
import torch.nn.functional as F


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

        # --- 데이터셋 타입 감지 및 로드 로직 ---
        """
        # 1. KITTI-360 감지
        if os.path.exists(os.path.join(args.source_path, "data_3d_raw")):
            dataset_type = 'KITTI360'
            base_path = args.source_path
            image_path = base_path
            start_indices = [897, 2907, 86, 10948, 2743]
            scene_ids = ['1', '2', '3', '4', '5']
            
            seq_idx = args.data_seq if args.data_seq is not None else 0
            start_index = start_indices[seq_idx]
            scene_id = scene_ids[seq_idx]
            
            cam_id = args.cam_id
            scene_info_list, all_lidar_points = sceneLoadTypeCallbacks["KITTI360"](root_path = args.source_path, root_image_path = image_path, sequence=scene_id,
                                                                                cam_id=cam_id, start_index=start_index,
                                                                                segment_length=30)
        """
        # 2. [수정됨] KITTI-RAW 감지 (oxts 폴더 존재 여부로 확인)
        if os.path.exists(os.path.join(args.source_path, "oxts")) or "KITTI_RAW" in args.source_path or "drive" in args.source_path:
            dataset_type = 'KITTI_RAW'
            print("Detected KITTI RAW dataset structure.")
            
            # 사용자 입력 경로: .../KITTI_RAW/2011_09_28/2011_09_28_drive_0002_sync
            data_path = args.source_path
            
            try:
                # 1. 경로 정규화 (끝에 붙은 / 제거 등)
                norm_path = os.path.normpath(data_path)
                drive_folder_name = os.path.basename(norm_path)  # 2011_09_28_drive_0002_sync
                
                # 2. 날짜 추출 (앞 10자리: 2011_09_28)
                date_str = drive_folder_name[:10]
                
                # 3. 상위 폴더 탐색을 통한 Calib 경로 자동 추정
                # 구조 가정:
                # Root/
                #   2011_09_28/
                #      2011_09_28_drive_0002_sync/ (현재 data_path)
                #      2011_09_28_calib/           (목표 calib_path - 보통 날짜 폴더 옆이나 안에 있음)
                
                # 현재 경로: .../2011_09_28/2011_09_28_drive_0002_sync
                parent_dir = os.path.dirname(norm_path)        # .../2011_09_28 (날짜 폴더)
                grandparent_dir = os.path.dirname(parent_dir)  # .../KITTI_RAW  (루트)

                # 캘리브레이션 폴더 후보군 탐색
                calib_candidates = [
                    os.path.join(grandparent_dir, date_str, f"{date_str}_calib"), # 1. 날짜 폴더 옆 형제 폴더
                    os.path.join(parent_dir, f"{date_str}_calib"),                # 2. 날짜 폴더 내부
                    os.path.join(grandparent_dir, f"{date_str}_calib")            # 3. 루트 바로 아래
                ]
                
                calib_path = None
                for candidate in calib_candidates:
                    if os.path.exists(candidate):
                        calib_path = candidate
                        break
                
                # 못 찾았으면 에러 대신 경고 출력 후, 데이터 경로의 상위 폴더를 임시로 지정 (Loader가 내부에서 찾도록 유도)
                if calib_path is None:
                    print(f"Warning: Could not auto-detect specific calib folder for date {date_str}.")
                    print(f"Checked: {calib_candidates}")
                    print("Assuming calib files are in the parent directory.")
                    calib_path = parent_dir 

                print(f"Auto-configured Paths:\n - Data: {data_path}\n - Calib: {calib_path}")
                
                target_cam_id = args.cam_id if args.cam_id is not None else "02"
                
                # dataset_readers_raw.py 호출
                scene_info_list, all_lidar_points = readKittiRawSceneInfo(
                    calib_path=calib_path, 
                    data_path=data_path, 
                    cam_id=target_cam_id,
                    start_index=args.start_index,   # 추가
                    segment_length=args.segment_length # 추가
                )
                
            except Exception as e:
                print(f"Error initializing KITTI RAW: {e}")
                import traceback
                traceback.print_exc()
                exit(1)

        else:
            assert False, "Could not recognize scene type! (Checked for 'data_3d_raw' or 'oxts')"

        self.scene_info_list = scene_info_list
        self.all_lidar_points = all_lidar_points

        if not self.loaded_iter:
            with open(scene_info_list.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info_list.test_cameras:
                camlist.extend(scene_info_list.test_cameras)
            if scene_info_list.train_cameras:
                camlist.extend(scene_info_list.train_cameras)
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


        #Generate flow model, the resolution should be the finest one
        self.generate_optical_flow()

        initialized_pose_cl = None
        if dataset_type == 'KITTI360' or dataset_type == 'KITTI' or dataset_type == 'KITTI_RAW':
            initial_rotation_mat = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            initialized_pose_cl = np.eye(4)
            initialized_pose_cl[:3,:3]  = initial_rotation_mat

        # Generate trajectory
        self.sensor_trajectory = SensorTrajectories(
            lidar_rotations=scene_info_list.lidar_rotations,
            lidar_translations=scene_info_list.lidar_translations,
            lidar_timestamps=[0. for i in range(len(scene_info_list.lidar_rotations))],
            pose_cl=scene_info_list.pose_cl,
            initialized_pose_cl=initialized_pose_cl
        )

        for resolution_scale in resolution_scales:
            for i in range(len(self.train_cameras[resolution_scale])):
                cam = self.train_cameras[resolution_scale][i]
                cam.generate_gradient_and_pixel_masks()
            for i in range(len(self.test_cameras[resolution_scale])):
                cam = self.test_cameras[resolution_scale][i]
                cam.generate_gradient_and_pixel_masks()

        rotation_wl_list, translations_wl_list = self.sensor_trajectory.get_all_lidar_poses()
        projection_matrix = self.train_cameras[resolution_scales[self.current_level]][0].projection_matrix

        self.gaussians.create_from_pcd_list(pcd_list=scene_info_list.point_cloud_list,
                                            spatial_lr_scale= self.cameras_extent,
                                            rotations_wl_list= rotation_wl_list,
                                            translations_wl_list=translations_wl_list,
                                            projection_matrix = projection_matrix,
                                            resolution=self.voxel_resolutions[self.current_level])

    def generate_optical_flow(self):
        resolution_scale = self.resolution_scales[-1]
        train_cameras = self.train_cameras[resolution_scale]

        dino_weights = torch.load(os.path.join(self.weight_path, 'dinov2_vitl14_pretrain.pth'))
        weights = torch.load(os.path.join(self.weight_path, 'roma_outdoor.pth'))
        H = train_cameras[0].image_height
        W = train_cameras[0].image_width
        flow_model = roma_outdoor(device='cuda', weights=weights, dinov2_weights=dino_weights, coarse_res=560,
                                       upsample_res=(H, W))

        def generate_optical_flow_one_group(flow_model, cameras, window_size = 2):
            cam_num = len(cameras)
            images = [(camera.original_image.detach().cpu().permute(1,2,0).numpy()*255.0).astype(np.uint8) for camera in cameras]
            all_flow_refs = {}
            all_flow_matchs = {}
            all_certainty_refs = {}
            all_certainty_matchs = {}
            print('Start to generate optical flow')
            for w in range(1, window_size+1):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('Window size', w)
                flow_refs = []
                flow_matchs = []
                certainty_refs = []
                certainty_matchs = []
                for i in range(cam_num-w):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print('Generating optical flow for camera %d' % i)
                    ref_image = images[i]
                    match_image = images[i+w]
                    warp, certainty = flow_model.match_image(ref_image, match_image, device='cuda')
                    flow_ref = warp[:, :W, 2:]
                    flow_match = warp[:, W:, :2]
                    certainty_ref = certainty[:, :W]
                    certainty_match = certainty[:, W:]
                    flow_refs.append(flow_ref.detach().cpu())
                    flow_matchs.append(flow_match.detach().cpu())
                    certainty_refs.append(certainty_ref.detach().cpu())
                    certainty_matchs.append(certainty_match.detach().cpu())
                all_flow_refs[w] = flow_refs
                all_flow_matchs[w] = flow_matchs
                all_certainty_refs[w] = certainty_refs
                all_certainty_matchs[w] = certainty_matchs

            return all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs

        window_size = 1
        all_flow_refs, all_flow_matchs, all_certainty_refs, all_certainty_matchs = generate_optical_flow_one_group(flow_model, train_cameras, window_size=window_size)
        del flow_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for resolution_scale in list(self.train_cameras.keys()):
            cameras = self.train_cameras[resolution_scale]
            level_width, level_height = cameras[0].image_width, cameras[0].image_height
            for w in range(1, window_size+1):
                flow_refs = all_flow_refs[w]
                flow_matchs = all_flow_matchs[w]
                certainty_refs = all_certainty_refs[w]
                certainty_matchs = all_certainty_matchs[w]

                flow_ref_level = [item.to(device = 'cuda') for item in flow_refs]
                flow_match_level = [item.to(device = 'cuda') for item in flow_matchs]
                certainty_ref_level = [item.to(device = 'cuda') for item in certainty_refs]
                certainty_match_level = [item.to(device = 'cuda') for item in certainty_matchs]

                for i in range(len(flow_refs)):
                    flow_ref, flow_match, certainty_ref, certainty_match = flow_ref_level[i], flow_match_level[i], certainty_ref_level[i], certainty_match_level[i]
                    flow_ref_level[i] = F.interpolate(flow_ref[None, :, :, :].permute(0,3,1,2), size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    flow_match_level[i] = F.interpolate(flow_match[None, :, :, :].permute(0,3,1,2), size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_ref_level[i] = F.interpolate(certainty_ref[None, None, :, :], size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)
                    certainty_match_level[i] = F.interpolate(certainty_match[None, None, :, :], size=(level_height, level_width),
                                                      mode='nearest').squeeze(0).permute(1,2,0)

                cam_num = len(cameras)
                assert (cam_num == (len(flow_ref_level)+w)) & (cam_num == (len(flow_match_level)+w)) & (cam_num == (len(certainty_ref_level)+w)) & (cam_num == (len(certainty_match_level)+w))
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
        return
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainImages(self, scale = 1.0):
        train_images = []
        for camera in self.train_cameras[scale]:
            image = camera.original_image.detach().cpu().permute(1,2,0).numpy()
            image *= 255
            train_images.append(image.astype(np.uint8))
        return train_images

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getTrainCameraByIndex(self, index, scale=1.0):
        return self.train_cameras[scale][index]