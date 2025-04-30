import os
# import anomalib.config as config # v2.0.0에서는 이 모듈이 다른 방식으로 사용되거나 구조가 변경되었습니다.
from anomalib.data import FolderDataset  # 경로는 동일할 수 있지만, 내부 클래스 구조는 다를 수 있습니다.
from anomalib.models import Patchcore # 모델 클래스 경로는 동일할 수 있습니다.
from anomalib.engine import Engine # Engine 클래스 경로는 동일할 수 있습니다.
# from anomalib.deploy import OpenVINOInferencer, TorchInferencer # 배포 모듈은 필요에 따라 별도 임포트 및 사용
import torch
import cv2
from PIL import Image
import numpy as np

# v2.0.0에서는 설정 클래스를 직접 임포트하여 사용할 수도 있습니다.
# 예: from anomalib.config.model import PatchcoreConfig
#     from anomalib.config.data import FolderDatasetConfig
#     from anomalib.config.trainer import TrainerConfig
#     from anomalib.config import get_configurable_config # 이 함수는 다른 방식으로 사용될 수 있습니다.

# 데이터셋 경로 설정
DATASET_PATH = "./dataset/apple"  # <<<< 이 부분을 실제 경로로 수정하세요

# 학습 설정 (Python 딕셔너리 사용)
# v2.0.0에서는 이 딕셔너리를 직접 사용하거나,
# 새로운 설정 객체로 변환하여 사용하게 됩니다.
# 여기서는 딕셔너리에서 직접 필요한 값을 추출하여 사용합니다.
model_config = {
    "model": {
        "name": "patchcore",
        "backbone": "resnet18",  # 특징 추출에 사용할 백본 네트워크
        "pre_trained": True,
        "layers": ["layer1", "layer2", "layer3"],  # PatchCore가 특징을 추출할 레이어
        "input_size": [100, 100],  # 모델 입력 이미지 크기 (정사각형 권장)
        # v2.0.0 Patchcore 생성자에 patch_size 인자가 없을 수 있습니다.
        # "patch_size": 3,
        "coreset_sampling_ratio": 0.1,  # 메모리 뱅크 구축 시 사용할 코어셋 샘플링 비율
        "num_neighbors": 9,  # 이상 점수 계산 시 사용할 최근접 이웃 수
    },
    "dataset": {
        "name": "folder",
        "format": "folder",
        "root": DATASET_PATH,  # 데이터셋 루트 경로
        "category": "apple",
        "train_batch_size": 1,
        "test_batch_size": 1,
        "num_workers": 4,
        "image_size": [100, 100],  # 학습/테스트 중 이미지 리사이즈 크기
        "transform_config": {
            "train": None,
            "eval": None,
        },
        # v2.0.0에서는 normal_dir, abnormal_dir, test_dir가 dataset 설정 딕셔너리 안에
        # 경로 형태로 직접 전달되지 않고 FolderDataset 생성자에 인자로 전달될 가능성이 높습니다.
        "normal_dir": "train/good",
        "abnormal_dir": "test/abnormal",
        "test_dir": "test/good",
        # "mask_dir": "test/abnormal/ground_truth",
    },
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": 1,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        # v2.0.0 Trainer 설정은 PyTorch Lightning Trainer의 인자와 더 유사해졌을 수 있습니다.
        # callbacks, logger는 직접 객체로 전달해야 할 수도 있습니다.
        "callbacks": [], # 빈 리스트로 변경
        "logger": None, # None 유지 또는 실제 Logger 객체 전달
        "default_root_dir": "results", # 결과 저장 경로 설정 인자 (None이면 자동 생성)
        # 추가적인 Trainer 인자 필요 시 여기에 추가
    },
    "inference": {
        "save_images": False,
        "save_path": "results",
        "visualize": False,
    },
    "seed": 42,
}

# v2.0.0: 설정 딕셔너리에서 직접 값을 추출하여 객체 생성

# 데이터 모듈 설정
# FolderDataset 생성자 인자를 v2.0.0 문서에 맞게 확인하고 전달해야 합니다.
# 아래는 예상되는 v2.0.0 FolderDataset 생성자 인자 전달 방식입니다.
datamodule = FolderDataset(
    name="apple",
    root=model_config["dataset"]["root"],
    normal_dir="train/good",  # 직접 경로 문자열 전달
    abnormal_dir="test/abnormal", # 직접 경로 문자열 전달
    normal_test_dir="test/good",  # 직접 경로 문자열 전달
    # image_size=model_config["dataset"]["image_size"],
    # train_batch_size=model_config["dataset"]["train_batch_size"],
    # test_batch_size=model_config["dataset"]["test_batch_size"],
    # num_workers=model_config["dataset"]["num_workers"],
    # transform_config=model_config["dataset"]["transform_config"],
    # v2.0.0 FolderDataset에 따라 mask_dir 인자를 추가해야 할 수 있습니다.
    # mask_dir=model_config["dataset"].get("mask_dir"), # .get()으로 안전하게 접근
    # mask_suffix=model_config["dataset"].get("mask_suffix"),
)
# datamodule.setup()  # 데이터셋 로딩 및 분할 준비

# 모델 초기화
# Patchcore 생성자 인자를 v2.0.0 문서에 맞게 확인하고 전달해야 합니다.
# 아래는 예상되는 v2.0.0 Patchcore 생성자 인자 전달 방식입니다.
model = Patchcore(
    # input_size=model_config["model"]["input_size"],
    backbone=model_config["model"]["backbone"],
    pre_trained=model_config["model"]["pre_trained"],
    layers=model_config["model"]["layers"],
    coreset_sampling_ratio=model_config["model"]["coreset_sampling_ratio"],
    num_neighbors=model_config["model"]["num_neighbors"],
    # v2.0.0 Patchcore에 다른 필수 인자가 있다면 추가해야 합니다.
)

# 학습 엔진 초기화
# Engine 생성자 인자를 v2.0.0 PyTorch Lightning Trainer 인자에 가깝게 전달합니다.
# model_config["trainer"] 딕셔너리를 unpack (**)하여 전달하는 방식이 v2.0.0에서도 유효할 수 있습니다.
# 하지만 정확한 인자 이름은 확인이 필요합니다.
# 예시: Engine(accelerator='auto', devices=1, max_epochs=1, ...)
engine_params = model_config["trainer"].copy() # Trainer 설정 딕셔너리 복사
# Trainer에 직접 전달해야 하는 객체(callbacks, logger)가 있다면 여기서 처리
# engine_params["callbacks"] = ...
# engine_params["logger"] = ...

engine = Engine(**engine_params)


print("=== 모델 학습 시작 ===")
# 모델 학습 (정상 데이터만 사용)
# engine.train() 메소드 사용법은 v2.0.0에서도 유사할 가능성이 높습니다.
engine.train(
    datamodule=datamodule,
    model=model,
)
print("=== 모델 학습 완료 ===")


print("=== 모델 테스트 시작 ===")
# 모델 테스트 (정상 및 불량 데이터 모두 사용)
# engine.test() 메소드 사용법은 v2.0.0에서도 유사할 가능성이 높으며, 결과 반환도 유사할 수 있습니다.
test_results = engine.test(
    datamodule=datamodule,
    model=model,
)
print("=== 모델 테스트 완료 ==케이")

# 테스트 결과 확인 (예시)
# v2.0.0에서 반환되는 결과 딕셔너리의 키 이름은 이전 버전과 다를 수 있습니다.
print("\n=== 테스트 결과 요약 ===")
print(test_results) # 테스트 결과 딕셔너리 출력

# 저장된 모델 체크포인트 경로
# v2.0.0에서 체크포인트 저장 경로는 PyTorch Lightning의 기본 동작을 따릅니다.
# engine.trainer.default_root_dir 값 등을 확인하여 경로를 구성할 수 있습니다.
# 정확한 경로는 실행 환경 및 설정에 따라 달라집니다.
# v2.0.0에서는 results 폴더가 기본일 수도 있습니다.
# 예: results/patchcore/version_X/checkpoints/last.ckpt
# 이 부분은 실제 v2.0.0 실행 후 생성되는 경로를 확인하고 수정하는 것이 좋습니다.
DEFAULT_ROOT_DIR = engine.trainer.default_root_dir or "results" # default_root_dir이 None이면 results 사용
# logger가 None인 경우 version이 0이 아닐 수 있습니다. 실제 로그를 확인하세요.
# version = engine.trainer.logger.version if engine.trainer.logger else 0 # 로거 없으면 버전 0으로 가정
# CHECKPOINT_PATH = os.path.join(DEFAULT_ROOT_DIR, model_config["model"]["name"], f"version_{version}", "checkpoints", "last.ckpt")

# 가장 최근 체크포인트 파일을 찾는 일반적인 방법 (경로 구조에 따라 수정 필요)
# 예시: default_root_dir/model_name/lightning_logs/version_*/checkpoints/last.ckpt
# 또는 default_root_dir/model_name/version_*/checkpoints/last.ckpt
# 정확한 구조는 실행 후 logs 또는 results 폴더를 확인하세요.
# 여기서는 engine 객체를 통해 가장 최근 또는 마지막 체크포인트 경로를 가져오는 방법을 시도합니다.
CHECKPOINT_PATH = None
if engine.trainer.checkpoint_callback:
     CHECKPOINT_PATH = engine.trainer.checkpoint_callback.last_model_path or engine.trainer.checkpoint_callback.best_model_path
     if CHECKPOINT_PATH:
          print(f"가장 최근/최상 체크포인트 경로: {CHECKPOINT_PATH}")

# 만약 위 방법으로 경로를 얻지 못했다면 수동으로 추정
if CHECKPOINT_PATH is None and (engine.trainer.default_root_dir or "results"):
     print("경고: 체크포인트 콜백에서 경로를 얻지 못했습니다. 기본 경로 구조로 추정합니다.")
     # 추정 경로는 실제 구조와 다를 수 있습니다. 실행 후 logs/results 폴더를 확인하세요.
     # 보통 default_root_dir/model_name/lightning_logs/version_*/checkpoints/last.ckpt 입니다.
     # 또는 default_root_dir/model_name/version_*/checkpoints/last.ckpt
     # 여기서는 간략하게 default_root_dir/model_name/version_0/checkpoints/last.ckpt를 시도합니다.
     estimated_path = os.path.join(DEFAULT_ROOT_DIR, model_config["model"]["name"], "version_0", "checkpoints", "last.ckpt")
     if os.path.exists(estimated_path):
         CHECKPOINT_PATH = estimated_path
         print(f"추정된 체크포인트 경로 존재: {CHECKPOINT_PATH}")


if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
    print(
        f"경고: 체크포인트 파일을 찾을 수 없습니다 ({CHECKPOINT_PATH}). 모델 로딩 및 추론이 불가능할 수 있습니다."
        "학습이 정상 완료되었는지 확인하고, 체크포인트 파일의 실제 경로를 찾아 CHECKPOINT_PATH를 수정하세요."
    )
    # 추론 예시를 실행하려면 CHECKPOINT_PATH가 반드시 필요합니다.
    # 체크포인트 파일이 없다면 여기서 종료하거나 추론 부분을 건너뛰어야 합니다.


# 4. 학습된 모델로 새로운 이미지 추론

print("\n=== 새로운 이미지 추론 예시 ===")

# 추론에 사용할 이미지 경로 (실제 파일 경로로 변경 필요)
NEW_IMAGE_PATH_NORMAL = "path/to/a/new/normal_apple.jpg"  # 정상 사과 이미지 예시
NEW_IMAGE_PATH_DEFECT = "path/to/a/new/defective_apple.jpg"  # 불량 사과 이미지 예시

# 학습된 모델 로드 (체크포인트 파일로부터)
# anomalib v2.0.0에서 모델 로드 방식은 PyTorch Lightning과 유사합니다.
# Model.load_from_checkpoint() 메소드를 사용하거나, 모델 구조 생성 후 state_dict 로드
try:
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"체크포인트 '{CHECKPOINT_PATH}'에서 모델 로드 시도 중...")
        # v2.0.0에서는 load_from_checkpoint 사용 시 config가 자동으로 로드될 수 있습니다.
        # 또는 모델 생성 시 사용했던 config 인자들을 load_from_checkpoint에 전달해야 할 수 있습니다.
        # Patchcore.load_from_checkpoint의 정확한 signature 확인 필요
        loaded_model = Patchcore.load_from_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            # load_from_checkpoint에 모델 생성 시 사용했던 인자들을 전달해야 할 수 있습니다.
            # 예: input_size=model_config["model"]["input_size"],
            #     backbone=model_config["model"]["backbone"],
            #     layers=model_config["model"]["layers"],
            #     coreset_sampling_ratio=model_config["model"]["coreset_sampling_ratio"],
            #     num_neighbors=model_config["model"]["num_neighbors"],
            # map_location='cpu' # 필요 시 주석 해제
        )

        # 모델을 평가 모드로 설정
        loaded_model.eval()
        # GPU 사용 시
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_model.to(device)
        print("모델 로드 및 장치 이동 완료.")


        # 추론 함수 정의
        def predict_single_image(model, image_path, device, image_size):
            if not os.path.exists(image_path):
                print(f"오류: 이미지 파일 '{image_path}'를 찾을 수 없습니다.")
                return None, None, None

            # 이미지 로드 및 전처리 (학습 시와 동일한 전처리 적용 필요)
            image = cv2.imread(image_path)
            if image is None:
                 print(f"오류: 이미지 파일 '{image_path}'를 읽을 수 없습니다.")
                 return None, None, None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
            image = Image.fromarray(image)
            # v2.0.0에서는 transform 파이프라인을 모델이나 Inferencer에서 직접 가져오거나 재구성할 수 있습니다.
            # 간단하게 수동 리사이즈 후 Tensor 변환 예시
            image = image.resize(image_size)  # 모델 입력 크기에 맞게 리사이즈
            image_np = np.array(image)

            # PIL Image to PyTorch Tensor (CHW, float, 0-1 범위)
            # Transform 파이프라인을 사용하는 것이 더 정확합니다.
            # FolderDataset에서 transform을 가져와 사용
            # transform = datamodule.test_data.dataset.transform # 학습 시 사용한 transform 가져오기 시도
            # if transform:
            #      image_tensor = transform(image_np) # numpy array -> CHW tensor
            # else: # transform을 가져오지 못하면 수동 변환
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0


            image_tensor = image_tensor.unsqueeze(0).to(device)  # 배치 차원 추가 및 장치 이동

            # 모델 추론
            # anomalib v2.0.0: predict 메소드 사용법은 유사할 수 있습니다.
            with torch.no_grad():
                # predict 메소드의 반환 값 구조는 v2.0.0 문서를 확인해야 합니다.
                # 이전 버전과 동일하게 'pred_scores', 'anomaly_maps'를 반환한다고 가정합니다.
                predictions = model.predict(image_tensor)

            # 결과 추출 (v2.0.0 결과 딕셔너리 키 이름 확인 필요)
            image_score = predictions.get("pred_scores", torch.tensor(-1.0)).item()  # 키 이름 확인 후 get() 사용
            anomaly_map = predictions.get("anomaly_maps", torch.zeros(image_size)).squeeze().cpu().numpy()  # 키 이름 확인 후 get() 사용

            return image_score, anomaly_map, image_np  # 원본 이미지 numpy 배열도 반환하여 시각화에 사용

        # 정상 이미지 추론
        print(f"\n'{NEW_IMAGE_PATH_NORMAL}' 추론 결과:")
        normal_score, normal_map, normal_img_np = predict_single_image(
            loaded_model, NEW_IMAGE_PATH_NORMAL, device, model_config["model"]["input_size"]
        )
        if normal_score is not None:
            print(f"  이미지 이상 점수: {normal_score:.4f}")
            # 이상 점수가 낮을 것으로 예상됩니다.
            # 시각화 코드는 이전과 동일하게 cv2 등을 사용할 수 있습니다.

        # 불량 이미지 추론
        print(f"\n'{NEW_IMAGE_PATH_DEFECT}' 추론 결과:")
        defect_score, defect_map, defect_img_np = predict_single_image(
            loaded_model, NEW_IMAGE_PATH_DEFECT, device, model_config["model"]["input_size"]
        )
        if defect_score is not None:
            print(f"  이미지 이상 점수: {defect_score:.4f}")
            # 이상 점수가 높을 것으로 예상됩니다.
            # 시각화 코드는 이전과 동일하게 cv2 등을 사용할 수 있습니다.

    else:
         print("\n체크포인트 파일이 유효하지 않아 추론을 건너뜀.")

except Exception as e:
    print(f"\n추론 중 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 오류 상세 정보 출력

# v2.0.0에서 모델 내보내기 방식도 변경되었을 수 있습니다.
# OpenVINO/TorchScript Inferencer 사용법은 v2.0.0 문서를 참고하세요.