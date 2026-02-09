# scene/dataset_readers_raw.py

import os
import sys
import numpy as np
import pykitti
from tqdm import tqdm
import json
from PIL import Image

from utils.graphics_utils import focal2fov
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
from plyfile import PlyData, PlyElement

# [핵심] 원본 클래스 가져오기
try:
    from scene.dataset_readers import CameraInfo, FrameSceneListInfo, SceneInfo
except ImportError:
    print("[Error] scene.dataset_readers Import Failed.")
    sys.exit(1)

def storePly(path, xyz, rgb):
    elements = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elements['x'] = xyz[:, 0]; elements['y'] = xyz[:, 1]; elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]; elements['green'] = rgb[:, 1]; elements['blue'] = rgb[:, 2]
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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
        W2C = np.eye(4); W2C[:3, :3] = cam.R; W2C[:3, 3] = cam.T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def readKittiRawSceneInfo(path, images, eval, llffhold=8, multiscale=False, cam_id="02", start_index=0, segment_length=50):
    path = os.path.normpath(path)
    date_str = os.path.basename(path)
    base_dir = os.path.dirname(path)

    print(f"\n[Init] KITTI RAW Loader | Date: {date_str}")

    if not os.path.exists(path): sys.exit(f"[Error] Path not found: {path}")
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    drive_folders = [f for f in subfolders if 'drive' in f and 'sync' in f]
    if not drive_folders: sys.exit(f"[Error] No 'drive_xxxx_sync' folder in {path}")
    
    target_drive = drive_folders[0]
    try: drive_id = target_drive.split('_drive_')[1].split('_sync')[0]
    except: drive_id = target_drive[-9:-5]
    print(f"[Auto] Target Drive: {target_drive} (ID: {drive_id})")

    dataset = pykitti.raw(base_dir, date_str, drive_id)
    print(f"[Load] PyKitti Loaded. Frames: {len(dataset)}")

    if cam_id == '00': K_full = dataset.calib.K_cam0; first_img = next(iter(dataset.cam0))
    elif cam_id == '01': K_full = dataset.calib.K_cam1; first_img = next(iter(dataset.cam1))
    elif cam_id == '02': K_full = dataset.calib.K_cam2; first_img = next(iter(dataset.cam2))
    else: K_full = dataset.calib.K_cam3; first_img = next(iter(dataset.cam3))
    
    K = K_full.copy()
    fx, fy = K[0, 0], K[1, 1]
    width, height = first_img.size

    all_points_world = []
    all_points_lidar = []
    all_lidar_rotations = []
    all_lidar_translations = []
    
    max_len = len(dataset)
    end_index = min(start_index + segment_length, max_len)
    
    raw_first_pose = dataset.oxts[start_index].T_w_imu.copy()
    shift_vec = raw_first_pose[:3, 3].copy()
    
    for i in tqdm(range(start_index, end_index), desc="LiDAR"):
        velo_points = dataset.get_velo(i); velo_xyz = velo_points[:, :3]
        T_w_imu = dataset.oxts[i].T_w_imu.copy()
        T_velo_imu = dataset.calib.T_velo_imu
        T_velo_world = T_w_imu @ T_velo_imu
        T_velo_world[:3, 3] -= shift_vec
        R = T_velo_world[:3, :3].copy()
        t = T_velo_world[:3, 3].copy()
        all_lidar_rotations.append(R)
        all_lidar_translations.append(t)
        pc_world = (R @ velo_xyz.T + t).T
        all_points_lidar.append(velo_xyz); all_points_world.append(pc_world)

    all_points_world = np.concatenate(all_points_world, axis=0)

    cam_infos = []; local_cam_infos = []
    
    if cam_id == '00': T_cam_imu = dataset.calib.T_cam0_imu
    elif cam_id == '01': T_cam_imu = dataset.calib.T_cam1_imu
    elif cam_id == '02': T_cam_imu = dataset.calib.T_cam2_imu
    else: T_cam_imu = dataset.calib.T_cam3_imu
    T_imu_cam_inv = np.linalg.inv(T_cam_imu)

    for i in tqdm(range(start_index, end_index), desc="Camera"):
        T_w_imu_raw = dataset.oxts[i].T_w_imu.copy()
        T_w_imu_shifted = T_w_imu_raw.copy()
        T_w_imu_shifted[:3, 3] -= shift_vec
        c2w = T_w_imu_shifted @ T_imu_cam_inv
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T.copy()
        T = w2c[:3, 3].copy()
        
        img_path = os.path.join(dataset.data_path, f"image_{cam_id}", 'data', f"{i:010d}.png")
        FovY, FovX = focal2fov(fy, height), focal2fov(fx, width)

        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image_path=img_path, image_name=os.path.basename(img_path), width=width, height=height, is_test=False, K=K, D=np.zeros(5))
        cam_infos.append(cam_info); local_cam_infos.append([cam_info])

    ply_folder = os.path.join(path, target_drive, 'ply')
    os.makedirs(ply_folder, exist_ok=True)
    seq_name = f"{date_str}_{drive_id}"
    
    shs = np.random.random((len(all_points_world), 3)) / 255.0
    storePly(os.path.join(ply_folder, "raw_global.ply"), all_points_world, SH2RGB(shs)*255)

    pcd_list = []; ply_path_list = []
    for idx, pc in enumerate(all_points_lidar):
        p_path = os.path.join(ply_folder, f"raw_{idx}.ply")
        storePly(p_path, pc, SH2RGB(np.random.random((len(pc), 3))/255.0)*255)
        pcd_list.append(fetchPly(p_path)); ply_path_list.append(p_path)

    cameras_data = []
    for cam in cam_infos:
        pos = -cam.R.T @ cam.T
        cameras_data.append({"id": int(cam.uid), "img_name": cam.image_name, "width": int(width), "height": int(height), "fx": float(fx), "fy": float(fy), "position": pos.tolist(), "rotation": cam.R.tolist()})
    with open(os.path.join(path, target_drive, f"cameras_{seq_name}.json"), "w") as f: json.dump(cameras_data, f, indent=4)

    frame_scene_list = FrameSceneListInfo(
        point_cloud_list=pcd_list, ply_path_list=ply_path_list, ply_path=os.path.join(ply_folder, "raw_global.ply"),
        lidar_rotations=all_lidar_rotations, lidar_translations=all_lidar_translations,
        train_cameras_list=local_cam_infos, test_cameras_list=local_cam_infos,
        train_cameras=cam_infos, test_cameras=cam_infos,
        nerf_normalization=getNerfppNorm(cam_infos), is_nerf_synthetic=False, pose_cl=dataset.calib.T_velo_imu
    )

    sceneLoadTypeCallbacks = {"KITTI_RAW": readKittiRawSceneInfo}
    return frame_scene_list, all_points_world