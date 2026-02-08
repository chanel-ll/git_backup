#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.


import struct
from collections import defaultdict
import os
import sys
import json
import numpy as np
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import focal2fov
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# ================================================================================
# [1] 구조체 및 헬퍼 함수 (dataset_readers.py와 100% 동일 유지)
# ================================================================================

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    K: np.array
    D: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

class FrameSceneListInfo(NamedTuple):
    point_cloud_list: list
    ply_path_list: list
    ply_path: str
    lidar_rotations: list
    lidar_translations: list
    train_cameras_list: list
    test_cameras_list: list
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    is_nerf_synthetic: bool
    pose_cl: np.ndarray

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = np.eye(4)
        W2C[:3, :3] = cam.R.T
        W2C[:3, 3] = cam.T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# ================================================================================
# [2] KITTI RAW Loader (중첩된 폴더 구조 고정 처리)
# ================================================================================

class KittiRawLoader:
    def __init__(self, calib_path, data_path, user_cam_id="00"):
        # 입력받은 경로: .../2011_09_28_calib 또는 .../2011_09_28_drive_0002_sync
        
        # 1. 날짜 추출 (폴더명 앞 10자리)
        # 예: 2011_09_28_drive_0002_sync -> 2011_09_28
        try:
            folder_name_data = os.path.basename(os.path.normpath(data_path))
            date_str = folder_name_data[:10] 
        except:
            date_str = "2011_09_28" # 기본값 혹은 에러 처리
            
        print(f"[Loader] Initial Paths:\n  Calib Input: {calib_path}\n  Data Input: {data_path}")

        # 2. Calibration 경로 고정 (중첩 구조 반영)
        # 구조: [calib_path] / [date_str] / calib_cam_to_cam.txt
        # 예: .../2011_09_28_calib / 2011_09_28 / calib_cam_to_cam.txt
        nested_calib_path = os.path.join(calib_path, date_str)
        
        if os.path.exists(os.path.join(nested_calib_path, "calib_cam_to_cam.txt")):
            self.calib_folder = nested_calib_path
        elif os.path.exists(os.path.join(calib_path, "calib_cam_to_cam.txt")):
            self.calib_folder = calib_path # 중첩이 없는 경우
        else:
            raise FileNotFoundError(f"Calib file not found in {nested_calib_path} or {calib_path}")

        # 3. Data 경로 고정 (중첩 구조 반영)
        # 구조: [data_path] / [date_str] / [folder_name_data] / oxts
        # 예: .../2011_09_28_drive_0002_sync / 2011_09_28 / 2011_09_28_drive_0002_sync / oxts
        nested_data_path = os.path.join(data_path, date_str, folder_name_data)
        
        if os.path.exists(os.path.join(nested_data_path, "oxts")):
            self.data_folder = nested_data_path
        elif os.path.exists(os.path.join(data_path, "oxts")):
            self.data_folder = data_path # 중첩이 없는 경우
        else:
            raise FileNotFoundError(f"Data folders (oxts) not found in {nested_data_path} or {data_path}")

        print(f"[Loader] Resolved Paths:\n  Calib: {self.calib_folder}\n  Data: {self.data_folder}")
        
        self.cam_id = user_cam_id
        
        # 4. 파일 로드 시작
        self.calib = self._load_calibration()
        
        self.oxts_path = os.path.join(self.data_folder, "oxts", "data")
        self.all_poses = self._load_oxts_and_convert()
        
        self.image_path = os.path.join(self.data_folder, f"image_{self.cam_id}", "data")
        self.velo_path = os.path.join(self.data_folder, "velodyne_points", "data")
        
        if not os.path.exists(self.image_path):
             raise FileNotFoundError(f"Image folder not found at: {self.image_path}")

        self.shift_translation = np.zeros(3)

    def _read_calib_file(self, filepath):
        data = {}
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calib file missing: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                if ':' not in line: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def _load_calibration(self):
        calib = {}
        cam2cam_path = os.path.join(self.calib_folder, "calib_cam_to_cam.txt")
        c2c_data = self._read_calib_file(cam2cam_path)
        
        P_rect = c2c_data[f'P_rect_{self.cam_id}'].reshape(3, 4)
        R_rect = c2c_data[f'R_rect_{self.cam_id}'].reshape(3, 3)
        K = P_rect[:3, :3]
        
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R_rect
        calib['R0_rect'] = R0_rect_4x4
        calib['K'] = K
        calib['D'] = np.zeros(5) 

        # Extrinsics
        velo2cam_path = os.path.join(self.calib_folder, "calib_velo_to_cam.txt")
        v2c_data = self._read_calib_file(velo2cam_path)
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[:3, :3] = v2c_data['R'].reshape(3,3)
        Tr_velo_to_cam[:3, 3] = v2c_data['T']
        calib['Tr_velo_to_cam'] = Tr_velo_to_cam

        imu2velo_path = os.path.join(self.calib_folder, "calib_imu_to_velo.txt")
        i2v_data = self._read_calib_file(imu2velo_path)
        Tr_imu_to_velo = np.eye(4)
        Tr_imu_to_velo[:3, :3] = i2v_data['R'].reshape(3,3)
        Tr_imu_to_velo[:3, 3] = i2v_data['T']
        calib['Tr_imu_to_velo'] = Tr_imu_to_velo

        calib['T_imu_to_cam'] = R0_rect_4x4 @ Tr_velo_to_cam @ Tr_imu_to_velo
        return calib

    def _load_oxts_and_convert(self):
        oxts_files = sorted(os.listdir(self.oxts_path))
        poses = []
        lat0, lon0, alt0 = None, None, None
        
        def latlon_to_mercator(lat, lon, scale):
            er = 6378137. 
            mx = scale * lon * np.pi * er / 180
            my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) )
            return mx, my

        for f_name in oxts_files:
            with open(os.path.join(self.oxts_path, f_name), 'r') as f:
                vals = [float(x) for x in f.read().split()]
                lat, lon, alt = vals[0], vals[1], vals[2]
                roll, pitch, yaw = vals[3], vals[4], vals[5]
                
                if lat0 is None:
                    lat0, lon0, alt0 = lat, lon, alt
                    scale = np.cos(lat0 * np.pi / 180.0)
                
                mx, my = latlon_to_mercator(lat, lon, scale)
                mx0, my0 = latlon_to_mercator(lat0, lon0, scale)
                
                tx = mx - mx0
                ty = my - my0
                tz = alt - alt0
                
                Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
                Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
                Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                
                R = Rz @ Ry @ Rx
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = [tx, ty, tz]
                poses.append(pose)
        return poses

    def get_image_path(self, idx):
        return os.path.join(self.image_path, f"{idx:010d}.png")

    def load_pointcloud(self, idx):
        pc_path = os.path.join(self.velo_path, f"{idx:010d}.bin")
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]
        points = points[np.logical_or(((np.abs(points[:, 0]) >= 1) & (np.abs(points[:, 1]) >= 1) & (points[:, 2] > -3)),
                                      ((points[:, 2] < -1)) & (points[:, 2] > -3))]
        return points

    def get_imu_pose(self, idx):
        pose = self.all_poses[idx].copy()
        pose[:3, 3] -= self.shift_translation
        return pose

    def get_camera_pose(self, idx):
        T_w_i = self.get_imu_pose(idx)
        T_cam_imu = self.calib['T_imu_to_cam'] 
        T_imu_cam = np.linalg.inv(T_cam_imu)   
        pose = T_w_i @ T_imu_cam
        return pose

    def get_lidar_pose(self, idx):
        T_w_i = self.get_imu_pose(idx)
        T_velo_imu = np.linalg.inv(self.calib['Tr_imu_to_velo'])
        T_imu_velo = np.linalg.inv(T_velo_imu)
        pose = T_w_i @ T_imu_velo
        return pose

    def set_shift(self, shift):
        self.shift_translation = shift
        
    def get_K(self):
        return self.calib['K']
    
    def get_image_shape(self):
        img = Image.open(self.get_image_path(0))
        return img.width, img.height

    def get_lidar_to_cam_pose(self):
        return self.calib['R0_rect'] @ self.calib['Tr_velo_to_cam']

# ================================================================================
# [3] Main Entry Point
# ================================================================================

def readKittiRawSceneInfo(calib_path, data_path, cam_id='00', start_index=0, segment_length=30):
    
    # [DEBUG START] ========================================================
    print(f"\n{'='*60}")
    print(f"[DEBUG] readKittiRawSceneInfo 진입 확인")
    print(f" - Calib Path 입력값: {calib_path}")
    print(f" - Data Path 입력값 : {data_path}")
    
    # 1. 캘리브레이션 파일 확인 (calib_path 기준)
    required_calib = ['calib_cam_to_cam.txt', 'calib_imu_to_velo.txt', 'calib_velo_to_cam.txt']
    missing_calib = []
    print(f"[DEBUG] Calibration 파일 검사:")
    for f in required_calib:
        f_full = os.path.join(calib_path, f)
        if os.path.exists(f_full):
            print(f"   ✅ 발견: {f}")
        else:
            print(f"   ❌ 누락: {f} (위치: {f_full})")
            missing_calib.append(f)

    # 2. GPS/IMU 데이터 확인 (data_path 기준)
    # 보통 data_path 안에 oxts 폴더가 있어야 함
    oxts_path = os.path.join(data_path, 'oxts')
    if os.path.exists(oxts_path) and os.path.isdir(oxts_path):
        txt_count = len([x for x in os.listdir(oxts_path) if x.endswith('.txt')])
        oxts_data_path = os.path.join(oxts_path, 'data') # KITTI raw 구조 대응
        if txt_count == 0 and os.path.exists(oxts_data_path):
             txt_count = len([x for x in os.listdir(oxts_data_path) if x.endswith('.txt')])
        
        print(f"[DEBUG] Oxts(GPS) 폴더 검사:")
        print(f"   ✅ 폴더 발견: {oxts_path}")
        print(f"   📄 내부 파일 수: {txt_count}개")
        if txt_count == 0:
            print(f"   ⚠️ 경고: oxts 폴더는 있는데 txt 파일이 없습니다!")
    else:
        print(f"[DEBUG] Oxts(GPS) 폴더 검사:")
        print(f"   ❌ 폴더 미발견: {oxts_path}")
        print(f"      (이게 없으면 포즈가 전부 0으로 나옵니다)")

    print(f"{'='*60}\n")
    # [DEBUG END] ==========================================================
    loader_root = os.path.dirname(data_path)
    
    print(f"[FIX] KittiRawLoader 경로 수정: {data_path} -> {loader_root}")
    
    loader = KittiRawLoader(calib_path, loader_root, user_cam_id=cam_id)
    
    K = loader.get_K()
    fx, fy = K[0, 0], K[1, 1]
    width, height = loader.get_image_shape()
    Tcl = loader.get_lidar_to_cam_pose()

    print(f"Loading KITTI Raw from:\n - Calib: {loader.calib_folder}\n - Data: {loader.data_folder}")
    print("Generate initial point cloud...")

    all_points_world = []
    all_points_lidar = []
    all_lidar_rotations = []
    all_lidar_translations = []
    
    max_len = len(loader.all_poses)
    end_index = min(start_index + segment_length, max_len)
    
    initial_translation = loader.get_lidar_pose(start_index)[:3, 3]
    loader.set_shift(initial_translation)

    # 1. LiDAR Loop
    for frame_id in tqdm(range(start_index, end_index)):
        pc_lidar = loader.load_pointcloud(frame_id)
        T_lidar = loader.get_lidar_pose(frame_id)
        R_lidar = T_lidar[:3, :3]
        t_lidar = T_lidar[:3, 3:4]
        
        all_lidar_rotations.append(R_lidar)
        all_lidar_translations.append(t_lidar.reshape(-1))
        
        pc_world = (R_lidar @ pc_lidar.T + t_lidar).T
        
        all_points_lidar.append(pc_lidar) 
        all_points_world.append(pc_world)
        
    all_points_world = np.concatenate(all_points_world, axis=0)

    # 2. Camera Loop
    cam_infos = []
    local_cam_infos = []
    
    for frame_id in tqdm(range(start_index, end_index), desc="Loading camera data"):
        c2w = loader.get_camera_pose(frame_id)
        w2c = np.linalg.inv(c2w)
        
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        img_path = loader.get_image_path(frame_id)
        
        cam_info = CameraInfo(
            uid=frame_id,
            R=R, T=T, FovY=FovY, FovX=FovX,
            image_path=img_path, image_name=os.path.basename(img_path),
            width=width, height=height, is_test=False, K=K, D=np.zeros(5)
        )
        cam_infos.append(cam_info)
        local_cam_infos.append([cam_info])

    nerf_norm = getNerfppNorm(cam_infos)

    # 3. PLY & JSON (dataset_readers.py 방식 엄수)
    if not os.path.exists(os.path.join(data_path, 'ply')):
        os.makedirs(os.path.join(data_path, 'ply'), exist_ok=True)
    
    seq_name = os.path.basename(os.path.normpath(data_path))

    # (A) Global PLY (Random SH)
    all_voxel_points_world = all_points_world
    shs = np.random.random((len(all_voxel_points_world), 3)) / 255.0
    colors = SH2RGB(shs) * 255
    global_ply_path = os.path.join(data_path, 'ply', f"raw_global_point_cloud_{seq_name}.ply")
    storePly(global_ply_path, all_voxel_points_world, colors)
    
    # (B) Per-Frame PLY (Save & Fetch)
    pcd_list = []
    ply_path_list = []
    
    for index, voxel_point_cloud in tqdm(enumerate(all_points_lidar), desc='Processing point cloud'):
        shs = np.random.random((len(voxel_point_cloud), 3)) / 255.0
        colors = SH2RGB(shs) * 255
        
        ply_path = os.path.join(data_path, 'ply', f"raw_{index}_point_cloud_{seq_name}.ply")
        storePly(ply_path, voxel_point_cloud, colors)
        
        pcd = fetchPly(ply_path)
        pcd_list.append(pcd)
        ply_path_list.append(ply_path)

    train_cameras = cam_infos
    test_cameras = cam_infos
    
    cameras_data = []
    for cam_info in cam_infos:
        rotation = cam_info.R.T
        position = cam_info.T
        position = -rotation.T @ position
        rotation = rotation.T
        
        camera_dict = {
            "id": int(cam_info.uid),
            "img_name": cam_info.image_name,
            "width": int(cam_info.width),
            "height": int(cam_info.height),
            "fx": float(fx),
            "fy": float(fy),
            "position": position.tolist(),
            "rotation": rotation.tolist()
        }
        cameras_data.append(camera_dict)

    cameras_json_path = os.path.join(data_path, f"cameras_{seq_name}.json")
    with open(cameras_json_path, "w", encoding="utf-8") as f:
        json.dump(cameras_data, f, indent=4, ensure_ascii=False)

    return FrameSceneListInfo(
        point_cloud_list=pcd_list,
        ply_path_list=ply_path_list,
        ply_path=global_ply_path,
        lidar_rotations=all_lidar_rotations,
        lidar_translations=all_lidar_translations,
        train_cameras=train_cameras,
        train_cameras_list=local_cam_infos,
        test_cameras=test_cameras,
        test_cameras_list=local_cam_infos,
        nerf_normalization=nerf_norm,
        is_nerf_synthetic=False,
        pose_cl=Tcl,
    ), all_points_world

print("Dataset_RAW 시작")
sceneLoadTypeCallbacks = {
    "KITTI_RAW": readKittiRawSceneInfo,
}
