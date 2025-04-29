import torch
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights, MobileNet_V2_Weights
import torchvision.transforms as transforms
import numpy as np
import os
import re
import shutil
import csv
import math
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import faiss
import pandas as pd

# ---- 기본 설정 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 3
TOP_K = 5
MEMORY_SAMPLING_RATIO = 0.1  # 메모리 뱅크 샘플링 비율 (10%)
IMAGE_SIZE = 224
OUTPUT_DIR = 'outputs'
OK_DIR = os.path.join(OUTPUT_DIR, 'ok')
NG_DIR = os.path.join(OUTPUT_DIR, 'ng')
CSV_FILE = os.path.join(OUTPUT_DIR, 'classification_result.csv')

# ---- 한글 폰트 설정 ----
mpl.rc('font', family='NanumGothic')  # 맑은 고딕
mpl.rcParams['axes.unicode_minus'] = False

# 폴더 준비
os.makedirs(OK_DIR, exist_ok=True)
os.makedirs(NG_DIR, exist_ok=True)

# ---- Transform 정의 ----
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
    if len(images) == 0:
        raise ValueError(f"No images found in {folder}. Please check the path and image files.")
    return torch.stack(images), filenames

# ---- 특징 추출기 ----
# Wide_ResNet50_2
# class FeatureExtractor(torch.nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.outputs = None
#         self._register_hook()

#     def _register_hook(self):
#         def hook_fn(module, input, output):
#             self.outputs = output
#         layer = dict([*self.backbone.named_modules()])['layer2']
#         layer.register_forward_hook(hook_fn)

#     def forward(self, x):
#         _ = self.backbone(x)
#         return self.outputs

# MobileNetV2
class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.features(x)
        return x

# ---- Patchify ----
def patchify(features, patch_size):
    B, C, H, W = features.shape
    features = features.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    patches = features.contiguous().view(B, -1, patch_size * patch_size * C)
    return patches

# ---- 모델 준비 ----
# backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
backbone = backbone.to(DEVICE).eval()
feature_extractor = FeatureExtractor(backbone)

# ---- 학습 데이터 준비 ----
train_imgs, _ = load_images('dataset/apple/train')
test_imgs, test_filenames = load_images('dataset/apple/test')

train_imgs = train_imgs.to(DEVICE)
test_imgs = test_imgs.to(DEVICE)

# ---- 특징 추출 ----
print("Extracting features...")
with torch.no_grad():
    train_features = feature_extractor(train_imgs)
    test_features = feature_extractor(test_imgs)

train_patches = patchify(train_features, PATCH_SIZE).view(-1, PATCH_SIZE * PATCH_SIZE * train_features.shape[1])
test_patches = patchify(test_features, PATCH_SIZE).view(test_imgs.shape[0], -1, PATCH_SIZE * PATCH_SIZE * test_features.shape[1])

# ---- Memory Bank 구성 ----
memory_bank = train_patches.cpu().numpy()
memory_bank = memory_bank[np.random.choice(len(memory_bank), int(len(memory_bank) * MEMORY_SAMPLING_RATIO), replace=False)]

# ---- faiss index 구축 ----
index = faiss.IndexFlatL2(memory_bank.shape[1])
index.add(memory_bank)

# ---- 이상 탐지 및 저장 ----
raw_scores = []
results = []
print("Detecting anomalies and saving outputs...")

for idx, (filename, test_patch) in enumerate(zip(test_filenames, test_patches)):
    test_patch_np = test_patch.cpu().numpy()
    D, _ = index.search(test_patch_np, TOP_K)
    anomaly_map = D.mean(axis=1)
    anomaly_score = anomaly_map.mean()
    raw_scores.append(anomaly_score)

# ---- Score log 변환 + 정규화 ----
log_scores = [math.log(s + 1) for s in raw_scores]
min_score = min(log_scores)
max_score = max(log_scores)
normalized_scores = [(s - min_score) / (max_score - min_score) for s in log_scores]

# ---- 자동 Threshold 추천 ----
auto_threshold = np.percentile(normalized_scores, 95)  # 상위 5% 기준으로 NG 판정

# ---- 결과 저장 및 분류 ----
ok_count = 0
ng_count = 0

for idx, (filename, test_patch) in enumerate(zip(test_filenames, test_patches)):
    test_patch_np = test_patch.cpu().numpy()
    D, _ = index.search(test_patch_np, TOP_K)
    anomaly_map = D.mean(axis=1)
    heatmap = anomaly_map.reshape(int(np.sqrt(len(anomaly_map))), -1)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    img = Image.open(os.path.join('dataset/apple/test', filename)).convert("RGB")
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    heatmap_resized = np.array(Image.fromarray(heatmap).resize((IMAGE_SIZE, IMAGE_SIZE)))
    ax[1].imshow(img)
    ax[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    ax[1].set_title(f"Anomaly Map\nScore: {normalized_scores[idx]:.3f}")
    ax[1].axis("off")

    plt.tight_layout()

    output_filename = f"{filename.split('.')[0]}_score{normalized_scores[idx]:.3f}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()

    label = 'OK' if normalized_scores[idx] <= auto_threshold else 'NG'
    dst_dir = OK_DIR if label == 'OK' else NG_DIR
    shutil.move(os.path.join(OUTPUT_DIR, output_filename), os.path.join(dst_dir, output_filename))

    results.append({
        'filename': output_filename,
        'score': normalized_scores[idx],
        'classification': label
    })

    if label == 'OK':
        ok_count += 1
    else:
        ng_count += 1

# ---- CSV 저장 ----
with open(CSV_FILE, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'score', 'classification']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# ---- 통계 그래프 그리기 ----
print("Generating statistics graphs...")
df = pd.read_csv(CSV_FILE)

labels = ['정상 (OK)', '이상 (NG)']
sizes = [ok_count, ng_count]
colors = ['#66b3ff', '#ff6666']

# Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('PatchCore 이상 탐지 통계 (Pie Chart)')
plt.savefig(os.path.join(OUTPUT_DIR, 'statistics_pie_chart.png'))
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(labels, sizes, color=colors)
plt.title('PatchCore 이상 탐지 통계 (Bar Chart)')
plt.ylabel('개수')
plt.savefig(os.path.join(OUTPUT_DIR, 'statistics_bar_chart.png'))
plt.show()

# ---- 최종 통계 출력 ----
print("✅ 최종 완료!")
print(f"정상(OK): {ok_count}개")
print(f"이상(NG): {ng_count}개")
print(f"자동 추천 Threshold: {auto_threshold:.3f}")
print(f"결과 CSV 저장 위치: {CSV_FILE}")

from fpdf import FPDF
import os

# ---- PDF 클래스 정의 ----
class PDF(FPDF):
    def header(self):
        korean_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        # self.add_font('NanumGothic', '', korean_font_path)
        # self.add_font('NanumGothic', 'B', korean_font_path)
        self.set_font('NanumGothic', 'B', 16)
        self.cell(0, 10, 'PatchCore 사과 불량 검출 리포트', ln=True, align='C')
        self.ln(10)

    def section_title(self, title):
        self.set_font('NanumGothic', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def section_body(self, body):
        self.set_font('NanumGothic', '', 11)
        self.multi_cell(0, 10, body)
        self.ln(5)

# ---- 리포트 생성 ----
def generate_pdf(ok_count, ng_count, auto_threshold, output_dir='outputs'):
    pdf = PDF()
    pdf.add_page()

    # 1. 기본 통계
    pdf.section_title('1. 통계 요약')
    summary = f'''
- 정상 (OK): {ok_count}개
- 이상 (NG): {ng_count}개
- 자동 추천 Threshold: {auto_threshold:.3f}
    '''
    pdf.section_body(summary)

    # 2. Pie Chart 삽입
    if os.path.exists(os.path.join(output_dir, 'statistics_pie_chart.png')):
        pdf.section_title('2. 정상/이상 비율 (Pie Chart)')
        pdf.image(os.path.join(output_dir, 'statistics_pie_chart.png'), x=30, w=150)
        pdf.ln(10)

    # 3. Bar Chart 삽입
    if os.path.exists(os.path.join(output_dir, 'statistics_bar_chart.png')):
        pdf.section_title('3. 정상/이상 개수 (Bar Chart)')
        pdf.image(os.path.join(output_dir, 'statistics_bar_chart.png'), x=30, w=150)
        pdf.ln(10)

    # 4. 대표 이미지 샘플 삽입
    pdf.section_title('4. 대표 결과 이미지')
    sample_dir = os.path.join(output_dir, 'ok')
    samples = sorted(os.listdir(sample_dir))[:2]  # OK 샘플 2장
    sample_dir_ng = os.path.join(output_dir, 'ng')
    samples_ng = sorted(os.listdir(sample_dir_ng))[:2]  # NG 샘플 2장

    for img_file in samples + samples_ng:
        if os.path.exists(os.path.join(sample_dir, img_file)):
            pdf.image(os.path.join(sample_dir, img_file), w=90)
            pdf.ln(5)
        elif os.path.exists(os.path.join(sample_dir_ng, img_file)):
            pdf.image(os.path.join(sample_dir_ng, img_file), w=90)
            pdf.ln(5)

    # 5. PDF 저장
    pdf.output(os.path.join(output_dir, 'patchcore_report.pdf'))
    print("✅ PDF 리포트 생성 완료! 저장 위치:", os.path.join(output_dir, 'patchcore_report.pdf'))

# ---- 사용 예시 ----
generate_pdf(ok_count, ng_count, auto_threshold)