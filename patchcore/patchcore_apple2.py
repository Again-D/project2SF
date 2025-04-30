import os
# import anomalib.config as config
# from anomalib.data import FolderDataset
# from anomalib.data import AnomalibDataModule
# from anomalib.data import FolderDataModule # 직접 임포트 필요 없음
from anomalib.data import get_datamodule # <-- get_datamodule 함수 임포트
from anomalib.pre_processing import PreProcessor

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.visualization import ImageVisualizer
from anomalib.metrics import F1Score, AUROC, Evaluator

import torch
import cv2
from PIL import Image
import numpy as np

# torchvision transforms 임포트
import torchvision.transforms.v2 as T

# ImageNet 데이터셋의 평균과 표준편차 (ResNet18 사전 학습에 사용됨)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 데이터셋 경로 설정
DATASET_PATH = "./dataset/apple"

# 학습 설정 (Python 딕셔너리 사용 - v2.0.0 구조에 맞춤)
model_config = {
    "model": {
        "name": "patchcore",
        "backbone": "resnet18",
        "pre_trained": True,
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.05,
        "num_neighbors": 9,
    },
    "dataset": {
        # get_datamodule 함수를 위한 설정 구조
        # class_path는 anomalib.data.__init__.py의 __all__ 목록에 있는 이름이어야 합니다.
        "class_path": "Folder", # <-- "anomalib.data.FolderDataModule" 대신 "Folder" 로 변경
        "init_args": { # <-- Folder 클래스 __init__ 인자에 정확히 맞춰야 합니다.
            "name": "apple_folder_dataset", # <-- Folder __init__ 필수 인자
            "root": DATASET_PATH, # <-- Folder __init__ 인자
            "normal_dir": "train/good", # <-- Folder __init__ 인자 (필수 아님)
            "abnormal_dir": "test/abnormal", # <-- Folder __init__ 인자
            "normal_test_dir": "test/good", # <-- "test_dir" 대신 "normal_test_dir"로 이름 변경
            # "mask_dir": "test/abnormal/ground_truth", # <-- Folder __init__ 인자 (필요 시 주석 해제)

            "train_batch_size": 1, # <-- Folder __init__ 인자
            "eval_batch_size": 1, # <-- "test_batch_size" 대신 "eval_batch_size"로 이름 변경
            "num_workers": 4, # <-- Folder __init__ 인자

            # Folder.__init__가 받는 augmentations 관련 인자들을 None으로 명시합니다.
            "train_augmentations": None,
            "val_augmentations": None,
            "test_augmentations": None,
            "augmentations": None,

            # Folder.__init__가 받는 split 관련 인자들도 필요 시 추가
            "test_split_mode": "from_dir",
            "test_split_ratio": 0.2,
            "val_split_mode": "from_test",
            "val_split_ratio": 0.5,
            "seed": 42, # seed 인자도 받습니다.
        }
    },
    "trainer": {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": 3,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "callbacks": [],
        "logger": None,
        "default_root_dir": "results",
    },
    "inference": {
        "save_images": True,
        "save_results": True,
        "show_results": False,
        "save_path": "results",
        "visualize": True,
    },
    "seed": 42,
}

# 1. get_datamodule 함수를 사용하여 DataModule 인스턴스화
# config 인자로 dataset 설정 딕셔너리 전체를 전달합니다.
datamodule = get_datamodule(config=model_config["dataset"])

# Initialize metrics with specific fields
f1_score = F1Score(fields=["pred_label", "gt_label"])
auroc = AUROC(fields=["pred_score", "gt_label"])
# Create evaluator with test metrics (for validation, use val_metrics arg)
evaluator = Evaluator(test_metrics=[f1_score, auroc])

transform = T.Compose([
    T.Resize((100, 100)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
])
pre_processor = PreProcessor(transform=transform)

visualizer = ImageVisualizer()

# 2. 모델 인스턴스화 (이 부분은 이전과 동일하게 유지)
try:
    model = Patchcore(
        # input_size=model_config["model"]["input_size"],
        backbone=model_config["model"]["backbone"],
        pre_trained=model_config["model"]["pre_trained"],
        layers=model_config["model"]["layers"],
        coreset_sampling_ratio=model_config["model"]["coreset_sampling_ratio"],
        num_neighbors=model_config["model"]["num_neighbors"],
        pre_processor=pre_processor,
        visualizer=visualizer,
        evaluator=evaluator,
    )
except TypeError as e:
    print(f"\nPatchcore 모델 생성 중 오류 발생: {e}")
    exit()

# 3. Engine 인스턴스화 (이 부분은 이전과 동일하게 유지)
try:
    engine = Engine(
        accelerator=model_config["trainer"]["accelerator"],
        devices=model_config["trainer"]["devices"],
        max_epochs=model_config["trainer"]["max_epochs"],
        default_root_dir=model_config["trainer"]["default_root_dir"],
        callbacks=model_config["trainer"]["callbacks"],
        logger=model_config["trainer"]["logger"],
    )
    if "seed" in model_config:
         import lightning as L
         L.seed_everything(model_config["seed"])

except TypeError as e:
    print(f"\nEngine 생성 중 오류 발생: {e}")
    exit()

print("=== 모델 학습 시작 ===")
engine.fit(
    datamodule=datamodule,
    model=model,
)
print("=== 모델 학습 완료 ===")

print("=== 모델 테스트 시작 ===")
test_results = engine.test(
    datamodule=datamodule,
    model=model,
)
print("=== 모델 테스트 완료 ===")

print("\n=== 테스트 결과 요약 ===")
print(test_results)

# ... (나머지 체크포인트 경로, 추론 코드 유지)
CHECKPOINT_PATH = None
if hasattr(engine.trainer, 'checkpoint_callback') and engine.trainer.checkpoint_callback:
     CHECKPOINT_PATH = engine.trainer.checkpoint_callback.last_model_path or engine.trainer.checkpoint_callback.best_model_path
     if CHECKPOINT_PATH:
          print(f"\n가장 최근/최상 체크포인트 경로: {CHECKPOINT_PATH}")
     else:
          print("\n체크포인트 콜백에서 경로를 얻지 못했습니다. 체크포인트 파일이 생성되었는지 확인하세요.")
else:
    print("\n체크포인트 콜백이 활성화되지 않았거나 접근할 수 없습니다. 체크포인트 파일이 저장되지 않았을 수 있습니다.")

if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
     print("경고: 체크포인트 파일을 찾을 수 없습니다. 기본 경로 구조로 추정하거나 수동으로 찾아야 합니다.")
     estimated_path = os.path.join(model_config["trainer"]["default_root_dir"], model_config["model"]["name"], "version_0", "checkpoints", "last.ckpt")
     if os.path.exists(estimated_path):
          CHECKPOINT_PATH = estimated_path
          print(f"추정된 체크포인트 경로 존재: {CHECKPOINT_PATH}")
     else:
          print("경고: 추정된 경로에서도 체크포인트 파일을 찾을 수 없습니다. 추론을 건너뜁니다.")
          CHECKPOINT_PATH = None

MODEL_PT_PATH = None # 새로 저장할 .pt 파일의 경로를 저장할 변수
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    try:
        print(f"'.ckpt' 파일을 로드하여 '.pt' 형식으로 다시 저장 시도 중...")
        # 2. .ckpt 파일을 로드합니다. Patchcore.load_from_checkpoint 메소드를 사용합니다.
        #    이 메소드는 .ckpt 형식을 로드할 수 있도록 설계되어 있습니다.
        #    모델 생성 시 사용했던 config를 load_from_checkpoint에 전달해야 할 수 있습니다.
        loaded_model_from_ckpt = Patchcore.load_from_checkpoint(
             checkpoint_path=CHECKPOINT_PATH,
             config=model_config["model"] # 모델 설정 딕셔너리를 config 인자로 전달
        )

        # 3. 새로 저장할 .pt 파일의 경로를 정의합니다.
        model_save_dir = os.path.join(model_config["trainer"]["default_root_dir"], model_config["model"]["name"])
        os.makedirs(model_save_dir, exist_ok=True) # 저장 폴더 생성
        MODEL_PT_PATH = os.path.join(model_save_dir, f"{model_config['model']['name']}_trained_model.pt") # .pt 확장자 사용

        # 4. 로드된 모델 객체 전체를 .pt 파일로 저장합니다.
        #    TorchInferencer가 load_model 메소드에서 기대하는 형식(모델 객체 또는 'model' 키를 가진 딕셔너리)
        #    에 맞추기 위해 모델 객체 전체를 저장합니다.
        model_to_save = {"model": loaded_model_from_ckpt}

        # 4. 'model' 키를 가진 딕셔너리를 .pt 파일로 저장합니다.
        torch.save(model_to_save, MODEL_PT_PATH)
        print(f"모델을 '.pt' 형식 (딕셔너리 구조)으로 저장 완료: {MODEL_PT_PATH}")
        

    except Exception as e:
        print(f"'.pt' 형식 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        MODEL_PT_PATH = None # 저장 실패 시 None으로 설정 유지

# Inferencer 및 추론 코드에서 사용할 체크포인트 경로를 새로 저장한 .pt 파일로 업데이트합니다.


print("\n=== 새로운 이미지 추론 예시 ===")

NEW_IMAGE_PATH_NORMAL = "./dataset/apple/test/good/60_100.jpg"
NEW_IMAGE_PATH_DEFECT = "./dataset/apple/test/abnormal/r0_11_100.jpg"

try:
    if MODEL_PT_PATH and os.path.exists(MODEL_PT_PATH):
        print(f"체크포인트 '{MODEL_PT_PATH}'에서 Inferencer 로드 시도 중...")
        
        inferencer = TorchInferencer(
            path=MODEL_PT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Inferencer 로드 완료.")

        print(f"\n'{NEW_IMAGE_PATH_NORMAL}' 추론 실행 중...")
        results_normal = inferencer.predict(image=NEW_IMAGE_PATH_NORMAL)
        print(f"  이미지 이상 점수 (Inferencer): {results_normal.pred_score.item():.4f}")
        print(f"  추론 결과 이미지가 '{model_config['inference']['save_path']}' 폴더에 저장되었습니다.")


        print(f"\n'{NEW_IMAGE_PATH_DEFECT}' 추론 실행 중...")
        results_defect = inferencer.predict(image=NEW_IMAGE_PATH_DEFECT)
        print(f"  이미지 이상 점수 (Inferencer): {results_defect.pred_score.item():.4f}")
        print(f"  추론 결과 이미지가 '{model_config['inference']['save_path']}' 폴더에 저장되었습니다.")

    else:
         print("\n체크포인트 파일이 유효하지 않아 추론을 건너뜀.")

except Exception as e:
    print(f"\n추론 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()