# debug_kitti_data.py
import os
import pykitti
import numpy as np

# 사용자 경로 설정
base_dir = '/local_datasets'
date = '2011_09_26'
drive = '0001'

print(f"{'='*50}")
print(f"[진단 시작] 경로: {base_dir}/{date}/{date}_drive_{drive}_sync")
print(f"{'='*50}")

# 1. 파일 존재 여부 확인 (Oxts)
oxts_path = os.path.join(base_dir, date, f"{date}_drive_{drive}_sync", "oxts", "data")
if not os.path.exists(oxts_path):
    print(f"❌ [CRITICAL] Oxts 폴더가 없습니다! 경로: {oxts_path}")
    exit()
else:
    files = sorted(os.listdir(oxts_path))
    print(f"✅ Oxts 폴더 발견. 파일 개수: {len(files)}개")
    if len(files) > 0:
        # 첫 번째 파일 내용 직접 읽기
        with open(os.path.join(oxts_path, files[0]), 'r') as f:
            print(f"📄 첫 번째 파일({files[0]}) 내용 일부:\n   -> {f.read().strip()[:50]}...")
        # 두 번째 파일 내용 직접 읽기 (값이 다른지 확인)
        with open(os.path.join(oxts_path, files[1]), 'r') as f:
            print(f"📄 두 번째 파일({files[1]}) 내용 일부:\n   -> {f.read().strip()[:50]}...")

# 2. PyKitti 로딩 테스트
print(f"\n[PyKitti 로딩 시도...]")
try:
    dataset = pykitti.raw(base_dir, date, drive)
    print(f"✅ PyKitti 로딩 성공. 총 프레임: {len(dataset)}")
except Exception as e:
    print(f"❌ PyKitti 로딩 실패: {e}")
    exit()

# 3. 포즈 값 변화 확인 (여기가 핵심)
print(f"\n[포즈 데이터 변동성 검사]")
pose0 = dataset.oxts[0].T_w_imu
pose1 = dataset.oxts[1].T_w_imu
pose10 = dataset.oxts[10].T_w_imu # 10번 프레임

print(f"▶ Frame 0 위치 (X,Y,Z): {pose0[:3, 3]}")
print(f"▶ Frame 1 위치 (X,Y,Z): {pose1[:3, 3]}")
print(f"▶ Frame 10 위치 (X,Y,Z): {pose10[:3, 3]}")

# 값이 변하는지 체크
diff = np.linalg.norm(pose0[:3, 3] - pose1[:3, 3])
if diff < 1e-6:
    print(f"\n❌ [문제 발견] Frame 0과 Frame 1의 위치가 똑같습니다! (Diff: {diff})")
    print("   -> 원인: Oxts 데이터가 모두 같은 값이거나, pykitti가 데이터를 제대로 파싱하지 못했습니다.")
else:
    print(f"\n✅ [정상] Frame 간 위치가 변하고 있습니다. (이동거리: {diff:.4f}m)")
    print("   -> 원인: 데이터는 정상인데, dataset_readers_raw.py의 변환 로직 문제일 가능성이 큽니다.")

print(f"{'='*50}")