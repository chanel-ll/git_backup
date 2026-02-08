# dataset_readers_raw.py (최종본)

import os
import sys
import numpy as np
import pykitti
from utils.graphics_utils import focal2fov
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
from tqdm import tqdm
from typing import NamedTuple
import json

# ==============================================================================
# [1] 구조체 정의
# ==============================================================================
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
    pose_cl: np.array

def storePly(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    elements = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def fetchPly(path):
    from plyfile import PlyData
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
        W2C = np.eye(4)
        W2C[:3, :3] = cam.R
        W2C[:3, 3] = cam.T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

# ==============================================================================
# [2] KITTI 로더 클래스 (PyKitti 활용)
# ==============================================================================
class KittiRawLoader:
    def __init__(self, basedir, date, drive, user_cam_id='02'):
        # basedir: /local_datasets
        # date: 2011_09_26
        # drive: 0001
        print(f"[Loader] PyKitti 초기화 시작: Base={basedir}, Date={date}, Drive={drive}")
        
        try:
            self.dataset = pykitti.raw(basedir, date, drive)
            print(f"[Loader] PyKitti 로딩 성공! Oxts 개수: {len(self.dataset.oxts)}")
        except Exception as e:
            print(f"[Loader] PyKitti 로딩 실패: {e}")
            print(f" -> 경로 확인: {basedir}/{date}/{date}_drive_{drive}_sync")
            sys.exit(1)

        self.user_cam_id = user_cam_id
        self.all_poses = [x for x in self.dataset.oxts]
        
    def get_K(self):
        if self.user_cam_id == '00': return self.dataset.calib.K_cam0
        if self.user_cam_id == '01': return self.dataset.calib.K_cam1
        if self.user_cam_id == '02': return self.dataset.calib.K_cam2
        if self.user_cam_id == '03': return self.dataset.calib.K_cam3
        return self.dataset.calib.K_cam2

    def get_image_shape(self):
        # cam2 기준 첫 번째 이미지 사이즈 반환
        return list(self.dataset.cam2)[0].size

    def get_lidar_to_cam_pose(self):
        return self.dataset.calib.T_velo_imu

    def get_lidar_pose(self, frame_id):
        return self.dataset.oxts[frame_id].T_w_imu

    def load_pointcloud(self, frame_id):
        return self.dataset.get_velo(frame_id)

    def get_camera_pose(self, frame_id):
        T_w_imu = self.dataset.oxts[frame_id].T_w_imu
        
        if self.user_cam_id == '00': T_cam_imu = self.dataset.calib.T_cam0_imu
        elif self.user_cam_id == '01': T_cam_imu = self.dataset.calib.T_cam1_imu
        elif self.user_cam_id == '02': T_cam_imu = self.dataset.calib.T_cam2_imu
        elif self.user_cam_id == '03': T_cam_imu = self.dataset.calib.T_cam3_imu
        else: T_cam_imu = self.dataset.calib.T_cam2_imu
        
        # C2W = T_w_imu @ inv(T_cam_imu)
        T_c2w = T_w_imu @ np.linalg.inv(T_cam_imu)
        return T_c2w

    def get_image_path(self, frame_id):
        img_folder = f"image_{self.user_cam_id}"
        # pykitti 내부 경로 활용
        return os.path.join(self.dataset.data_path, img_folder, 'data', f"{frame_id:010d}.png")

    def set_shift(self, shift):
        self.shift = shift

# ==============================================================================
# [3] 메인 함수: readKittiRawSceneInfo (경로 자동 탐색)
# ==============================================================================
def readKittiRawSceneInfo(path, images, eval, llffhold=8, multiscale=False, cam_id="02", start_index=0, segment_length=50):
    
    # 1. 입력 경로 파싱 (/local_datasets/2011_09_26 형태 가정)
    path = os.path.normpath(path)
    date_str = os.path.basename(path)       # 2011_09_26
    base_dir = os.path.dirname(path)        # /local_datasets

    print(f"\n{'='*60}")
    print(f"[Init] 입력 경로 분석:")
    print(f" - Base Dir: {base_dir}")
    print(f" - Date:     {date_str}")
    
    # 2. 캘리브레이션 파일 확인
    calib_files = ["calib_cam_to_cam.txt", "calib_imu_to_velo.txt", "calib_velo_to_cam.txt"]
    for cf in calib_files:
        if not os.path.exists(os.path.join(path, cf)):
            sys.exit(f"[Error] {path} 안에 {cf} 파일이 없습니다!")
    print(f"[Pass] 캘리브레이션 파일 확인 완료")

    # 3. 드라이브 폴더 자동 탐색
    if not os.path.exists(path):
        sys.exit(f"[Error] 경로 없음: {path}")

    # 'drive'와 'sync'가 들어간 폴더 찾기
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    drive_folders = [f for f in subfolders if 'drive' in f and 'sync' in f]
    
    if not drive_folders:
        sys.exit(f"[Error] {path} 안에 'drive_xxxx_sync' 폴더가 없습니다.")
    
    # 첫 번째 발견된 드라이브 사용 (예: 2011_09_26_drive_0001_sync)
    target_drive = drive_folders[0]
    
    # 'drive_0001' 부분에서 '0001' 추출
    # 예: 2011_09_26_drive_0001_sync -> split('_drive_')[1] -> 0001_sync -> split('_sync')[0] -> 0001
    try:
        drive_str = target_drive.split('_drive_')[1].split('_sync')[0]
    except:
        print(f"[Warn] 드라이브 폴더명 파싱 실패 ({target_drive}). 강제로 뒤에서 9~5번째 사용")
        drive_str = target_drive[-9:-5]

    print(f"[Auto] 드라이브 폴더 선택: {target_drive} (ID: {drive_str})")
    
    # 4. 로더 초기화
    loader = KittiRawLoader(base_dir, date_str, drive_str, user_cam_id=cam_id)
    
    K = loader.get_K()
    fx, fy = K[0, 0], K[1, 1]
    width, height = loader.get_image_shape()
    Tcl = loader.get_lidar_to_cam_pose()

    # 5. 데이터 로딩 루프
    all_points_world = []
    all_points_lidar = []
    all_lidar_rotations = []
    all_lidar_translations = []
    
    max_len = len(loader.all_poses)
    if start_index >= max_len:
        sys.exit(f"[Error] start_index({start_index})가 데이터 길이({max_len})보다 큽니다.")

    end_index = min(start_index + segment_length, max_len)
    
    print(f"[Run] {start_index} ~ {end_index} 프레임 로딩 시작...")
    initial_translation = loader.get_lidar_pose(start_index)[:3, 3]
    loader.set_shift(initial_translation)

    for frame_id in tqdm(range(start_index, end_index), desc="LiDAR"):
        pc_lidar = loader.load_pointcloud(frame_id)[:, :3]
        T_lidar = loader.get_lidar_pose(frame_id)
        R_lidar = T_lidar[:3, :3]
        t_lidar = T_lidar[:3, 3:4]
        
        all_lidar_rotations.append(R_lidar)
        all_lidar_translations.append(t_lidar.reshape(-1))
        pc_world = (R_lidar @ pc_lidar.T + t_lidar).T
        
        all_points_lidar.append(pc_lidar) 
        all_points_world.append(pc_world)
        
    all_points_world = np.concatenate(all_points_world, axis=0)

    # 6. 카메라 로딩
    cam_infos = []
    local_cam_infos = []
    
    for frame_id in tqdm(range(start_index, end_index), desc="Camera"):
        c2w = loader.get_camera_pose(frame_id)
        w2c = np.linalg.inv(c2w)
        R, T = w2c[:3, :3].T, w2c[:3, 3]
        
        FovY, FovX = focal2fov(fy, height), focal2fov(fx, width)
        img_path = loader.get_image_path(frame_id)
        
        cam_info = CameraInfo(
            uid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX,
            image_path=img_path, image_name=os.path.basename(img_path),
            width=width, height=height, is_test=False, K=K, D=np.zeros(5)
        )
        cam_infos.append(cam_info)
        local_cam_infos.append([cam_info])

    # 7. PLY 저장
    ply_folder = os.path.join(path, target_drive, 'ply')
    os.makedirs(ply_folder, exist_ok=True)
    seq_name = f"{date_str}_{drive_str}"
    
    # Global PLY
    shs = np.random.random((len(all_points_world), 3)) / 255.0
    storePly(os.path.join(ply_folder, f"raw_global.ply"), all_points_world, SH2RGB(shs)*255)

    # Per-frame
    pcd_list, ply_path_list = [], []
    for i, pc in enumerate(all_points_lidar):
        p_path = os.path.join(ply_folder, f"raw_{i}.ply")
        storePly(p_path, pc, SH2RGB(np.random.random((len(pc), 3))/255.0)*255)
        pcd_list.append(fetchPly(p_path))
        ply_path_list.append(p_path)

    return FrameSceneListInfo(
        point_cloud_list=pcd_list, ply_path_list=ply_path_list, ply_path=os.path.join(ply_folder, f"raw_global.ply"),
        lidar_rotations=all_lidar_rotations, lidar_translations=all_lidar_translations,
        train_cameras=cam_infos, train_cameras_list=local_cam_infos,
        test_cameras=cam_infos, test_cameras_list=local_cam_infos,
        nerf_normalization=getNerfppNorm(cam_infos), is_nerf_synthetic=False, pose_cl=Tcl
    ), all_points_world

sceneLoadTypeCallbacks = {"KITTI_RAW": readKittiRawSceneInfo}