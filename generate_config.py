import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_kitti360_configs(output_dir):
    # 1. Intrinsics [fx, fy, cx, cy]
    # data_rect 이미지를 쓰므로 P_rect_00 의 K값을 사용합니다.
    intrinsics_K = [552.554261, 552.554261, 682.049453, 238.769549] 
    intrinsics_D = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 2. Extrinsics GT 파싱 (calib_cam_to_velo.txt)
    cam2velo_vals = [
        0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
        -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
        -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824
    ]
    T_cam2velo = np.eye(4)
    T_cam2velo[:3, :4] = np.array(cam2velo_vals).reshape(3, 4)
    
    # LiDAR -> Unrectified Camera 역행렬 변환
    T_velo2cam_unrect = np.linalg.inv(T_cam2velo)
    R_velo2cam_unrect = T_velo2cam_unrect[:3, :3]
    t_velo2cam_unrect = T_velo2cam_unrect[:3, 3]

    # 🌟 [신규 추가] R_rect_00 (perspective.txt 기준) 반영
    # 이 행렬이 곱해져야 Rectified 이미지에 맞는 완벽한 정답(GT)이 됩니다.
    R_rect_vals = [
        0.999974, -0.007141, -0.000089,
        0.007141,  0.999969, -0.003247,
        0.000112,  0.003247,  0.999995
    ]
    R_rect_00 = np.array(R_rect_vals).reshape(3, 3)

    # 최종 GT Matrix = R_rect * T_velo2cam
    R_gt_final = R_rect_00 @ R_velo2cam_unrect
    t_gt_final = R_rect_00 @ t_velo2cam_unrect

    # Quaternion [w, x, y, z] 변환
    q_xyzw = R.from_matrix(R_gt_final).as_quat()
    gt_quat_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
    gt_trans = [float(t) for t in t_gt_final]

    # ==========================================
    # 3. gt.json 생성
    # ==========================================
    gt_data = {
        "intrinsics": {"K": intrinsics_K, "D": intrinsics_D},
        "extrinsics": {"translation": gt_trans, "rotation": gt_quat_wxyz}
    }
    with open(f"{output_dir}/gt.json", 'w') as f:
        json.dump(gt_data, f, indent=4)

    # ==========================================
    # 4. config.json 생성
    # ==========================================
    # 논문 저자 방식의 가혹한 노이즈 (회전 +10도, 이동 +0.2m)
    noisy_euler = R.from_quat(q_xyzw).as_euler('XYZ', degrees=True) + np.array([10.0, 10.0, 10.0])
    noisy_q_xyzw = R.from_euler('XYZ', noisy_euler, degrees=True).as_quat()
    noisy_quat_wxyz = [float(noisy_q_xyzw[3]), float(noisy_q_xyzw[0]), float(noisy_q_xyzw[1]), float(noisy_q_xyzw[2])]
    noisy_trans = [gt_trans[0] + 0.2, gt_trans[1] + 0.2, gt_trans[2] + 0.2]

    config_data = {
        "base_dir": output_dir,
        "frame_nums_per_batch": 3, # 🌟 1장에서 5장으로 증가 (안정적인 기하학적 제약 확보)
        "overlap_nums_between_batch": 0,
        "data_params": {
            "mono_depth_model": "depth_anything_v2",
            "half_resolution": False,
            "points_down_sample_step": 2, # 속도를 위해 샘플링 비율 조정
            "intensity_equalization": True,
            "gray_image_equalization": True,
            "shuffle": False
        },
        "pipeline_params": {
            "mode": 0,
            "patch_size": 40,
            "init_rot_range": 10.0,
            "init_rot_resolution": 1.0,
            "coarse_trans_range": 0.2,
            "coarse_iters": 300, # 🌟 150 -> 300 (충분한 탐색 시간 부여)
            "fine_trans_range": 0.2,
            "fine_iters": 300    # 🌟 150 -> 300
        },
        "intrinsics": {"K": intrinsics_K, "D": intrinsics_D},
        "extrinsics": {"translation": noisy_trans, "rotation": noisy_quat_wxyz}
    }

    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_data, f, indent=4)
        
    print(f"✅ 설정 파일 재생성 완료! (R_rect 반영 및 최적화 하이퍼파라미터 적용)")

if __name__ == "__main__":
    target_dir = "/home/airlab/claim/kitti360_dataset/kitti360_01" # 찬의님 경로로 맞춤
    generate_kitti360_configs(target_dir)