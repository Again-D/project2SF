import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# ---- 하이퍼파라미터 설정 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 3  # Patch 추출용
TOP_K = 5  # k-NN에서 비교할 최근접 이웃 수
THRESHOLD = 0.5  # 이상 판단 기준 (0~1)
FEATURE_LAYER = 'layer2'  # 특징 추출 레이어
IMAGE_SIZE = 224

# ---- 전처리 ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ---- 데이터 로드 함수 ----
def load_images(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)
            filenames.append(filename)
    return torch.stack(images), filenames

# ---- 특징 추출기 ----
class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layer_name):
        super().__init__()
        self.backbone = backbone
        self.layer_name = layer_name
        self.outputs = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.outputs = output
        layer = dict([*self.backbone.named_modules()])[self.layer_name]
        layer.register_forward_hook(hook_fn)

    def forward(self, x):
        _ = self.backbone(x)
        return self.outputs

# ---- 모델 준비 ----
backbone = models.wide_resnet50_2(pretrained=True)
backbone = backbone.to(DEVICE).eval()
feature_extractor = FeatureExtractor(backbone, FEATURE_LAYER)

# ---- 데이터 로드 ----
train_imgs, _ = load_images('dataset/apple/train')
test_imgs, test_filenames = load_images('dataset/apple/test')

train_imgs = train_imgs.to(DEVICE)
test_imgs = test_imgs.to(DEVICE)

# ---- 특징 추출 ----
print("Extracting features...")
with torch.no_grad():
    train_features = feature_extractor(train_imgs)
    test_features = feature_extractor(test_imgs)

# ---- Patch 단위로 나누기 ----
def patchify(features, patch_size):
    B, C, H, W = features.shape
    features = features.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    patches = features.contiguous().view(B, -1, patch_size * patch_size * C)
    return patches

train_patches = patchify(train_features, PATCH_SIZE).view(-1, PATCH_SIZE * PATCH_SIZE * train_features.shape[1])
test_patches = patchify(test_features, PATCH_SIZE).view(test_imgs.shape[0], -1, PATCH_SIZE * PATCH_SIZE * test_features.shape[1])

# ---- Memory Bank 구축 (Coreset) ----
memory_bank = train_patches.cpu().numpy()

# ---- k-NN 모델 준비 ----
print("Training k-NN...")
nn_model = NearestNeighbors(n_neighbors=TOP_K, metric='euclidean')
nn_model.fit(memory_bank)

# ---- 이상 탐지 ----
print("Detecting anomalies...")
scores = []
heatmaps = []

for test_patch in tqdm(test_patches):
    test_patch_np = test_patch.cpu().numpy()
    distances, _ = nn_model.kneighbors(test_patch_np)
    anomaly_map = distances.mean(axis=1)
    scores.append(anomaly_map.mean())
    heatmaps.append(anomaly_map.reshape(int(np.sqrt(len(anomaly_map))), -1))

scores = np.array(scores)

# ---- 결과 시각화 ----
os.makedirs('outputs', exist_ok=True)

for idx, (filename, heatmap) in enumerate(zip(test_filenames, heatmaps)):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # 원본 이미지
    img = Image.open(os.path.join('dataset/apple/test', filename)).convert("RGB")
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    # Heatmap
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((IMAGE_SIZE, IMAGE_SIZE)))
    ax[1].imshow(img)
    ax[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    ax[1].set_title(f"Anomaly Map\nScore: {scores[idx]:.3f}")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(f'outputs/{filename}_anomaly.png')
    plt.close()

print("완료! 결과는 'outputs/' 폴더에 저장되었습니다.")
