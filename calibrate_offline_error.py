import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pipeline import get_pipeline
from dataset.private_dataset import PrivateDataset
from utils.analyze_utils import analyze_results
from utils.vis_utils import draw_batch_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate with private dataset')
    parser.add_argument('--config', type=str, help='path of config file', required=True)
    parser.add_argument('--seed', type=int, help='seed for reproduction. Set a negetive number means using a random seed', default=525)
    parser.add_argument('--result_name', type=str, help='name of output result directory', default="results")
    parser.add_argument('--vis_proj', action='store_true', help='whether to visualize projection results')
    args = parser.parse_args()

    # load config
    assert os.path.exists(args.config) and args.config.endswith(".json")
    config = json.load(open(args.config))
    
    # build dataset
    dataset = PrivateDataset(
        base_dir = config["base_dir"],
        K = np.array(config["intrinsics"]["K"]),
        D = config["intrinsics"]["D"],
        **config["data_params"]
    )

    # get pipeline
    pipe = get_pipeline(config["pipeline_params"])

    # get initial guess
    T_init = np.eye(4)
    T_init[:3, :3] = R.from_quat(config["extrinsics"]["rotation"], scalar_first=True).as_matrix()
    T_init[:3, -1] = config["extrinsics"]["translation"]
    T_init = torch.from_numpy(T_init).float().cuda()
    
    # calib
    assert config["overlap_nums_between_batch"] <= config["frame_nums_per_batch"]
    
    # make result path
    result_path = os.path.join(config["base_dir"], args.result_name)
    if os.path.exists(result_path):
        shutil.rmtree(result_path, ignore_errors=True)
    os.makedirs(result_path, exist_ok=True)
    if args.vis_proj:
        proj_path = os.path.join(result_path, "proj")
        os.makedirs(proj_path, exist_ok=True)

    # set seed
    if args.seed > 0:
        torch.manual_seed(args.seed)

    results = []
    for i in range(0, len(dataset), config["frame_nums_per_batch"] - config["overlap_nums_between_batch"]):
    # for i in range(0, 5, config["frame_nums_per_batch"] - config["overlap_nums_between_batch"]):
        frames = []
        for k in range(i, min(i + config["frame_nums_per_batch"], len(dataset))):
            frames.append(dataset[k])
        
        # print frame infos
        print("*" * 50, "CLAIM", "*" * 50)
        for frame in frames:
            text = f"  🖼️   [Frame] {frame.frame_id}"
            print(f"\033[1;30;42m{text}{' '*(107 - len(text))}\033[0m")
        print("*" * 107)

        # calibration!
        T_est = pipe(T_init, frames)
        
        info = {
            "frame_id" : [frame.frame_id for frame in frames],
            "T_init" : T_init.cpu().numpy().tolist(),
            "T_est" : T_est.cpu().numpy().tolist()
        }
        results.append(info)

        # visualize projection
        if args.vis_proj:
            img_final = draw_batch_results(frames, T_init, T_est)
            img_name = '_'.join(info['frame_id'])
            cv2.imwrite(f"{proj_path}/{img_name}.jpg", img_final)
        
        
    # output result
    print(f"🖨️ Saving results to {result_path}")
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)
    
    # output analyzed results
    analyzed_results = analyze_results(results, vis=True)
    with open(os.path.join(result_path, "analyzed_results.png"), 'wb') as file:
        file.write(analyzed_results["fig"].getvalue())
    analyzed_results.pop("fig")
    with open(os.path.join(result_path, "analyzed_results.json"), "w") as f:
        json.dump(analyzed_results, f)
    
    # ==========================================================
    # 📊 Claim Paper Metrics (e_r, e_t^-) Evaluation
    # ==========================================================
    print("📊 Calculating Claim Paper Metrics...")
    
    # 1. 실제 GT 데이터 파일 로드 (config["base_dir"] 경로 활용)
    gt_path = os.path.join(config["base_dir"], "gt.json")
    if not os.path.exists(gt_path):
        print(f"⚠️ Warning: GT file not found at {gt_path}. Cannot calculate metrics.")
    else:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        gt_ext = gt_data['extrinsics']
        rot_list = gt_ext['rotation']
        
        # 이전 스크립트에서 확인한 [w, x, y, z] -> scipy [x, y, z, w] 매핑 유지
        gt_quat = np.array([rot_list[1], rot_list[2], rot_list[3], rot_list[0]])
        gt_t = np.array(gt_ext['translation'])

        # 2. 예측 데이터 추출 (메모리에 있는 analyzed_results 딕셔너리 활용)
        res_mean = analyzed_results['mean']
        est_euler_vec = np.array([
            res_mean['euler']['x'], 
            res_mean['euler']['y'], 
            res_mean['euler']['z']
        ])
        est_t = np.array([
            res_mean['translation']['x'], 
            res_mean['translation']['y'], 
            res_mean['translation']['z']
        ])

        # 3. 회전 행렬 및 오일러 변환
        gt_rot_obj = R.from_quat(gt_quat) # 매핑을 바꿨으므로 scalar_first 옵션 불필요
        gt_euler_vec = gt_rot_obj.as_euler('XYZ', degrees=True)
        gt_R = gt_rot_obj.as_matrix()

        est_R = R.from_euler('XYZ', est_euler_vec, degrees=True).as_matrix()

        # 4. 오차 계산 (Claim 논문 Equation 9)
        # er = |r_gt - r_est|
        e_r = np.linalg.norm(gt_euler_vec - est_euler_vec)

        # e-t = |(R_gt)^-1 * t_gt - (R_est)^-1 * t_est|
        gt_t_lidar = gt_R.T @ gt_t
        est_t_lidar = est_R.T @ est_t
        e_t_minus = np.linalg.norm(gt_t_lidar - est_t_lidar)

        # 5. 결과 출력
        print("\n" + "="*55)
        print(" 📊 [Claim Paper Calibration Metrics Evaluation]")
        print("="*55)
        print(f" Rotation Error (e_r)      : {e_r:.6f} (deg)")
        print(f" Translation Error (e_t^-) : {e_t_minus:.6f} (m)")
        print("="*55 + "\n")

    








    