import os
import re
import shutil
import csv

# ---- 설정 ----
OUTPUT_DIR = "outputs"
OK_DIR = os.path.join(OUTPUT_DIR, "ok")
NG_DIR = os.path.join(OUTPUT_DIR, "ng")
CSV_FILE = os.path.join(OUTPUT_DIR, "classification_result.csv")
THRESHOLD = 0.2  # 이상 탐지 기준점 (0~1 사이)

# 폴더 없으면 생성
os.makedirs(OK_DIR, exist_ok=True)
os.makedirs(NG_DIR, exist_ok=True)

# 정규표현식으로 파일명에서 Score 추출
score_pattern = re.compile(r"Score:\s*([\d\.]+)")

# 결과 저장용 리스트
results = []

# 분류 카운터
ok_count = 0
ng_count = 0

# outputs 폴더 내 파일 순회
for filename in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, filename)

    if filename.endswith(".png"):
        # 파일명에 score 정보가 없으면 무시
        if not "Score" in filename:
            continue

        # 파일명에서 Score 추출
        match = score_pattern.search(filename)
        if match:
            score = float(match.group(1))
            # 정상/이상 판정
            if score <= THRESHOLD:
                label = "OK"
                shutil.move(filepath, os.path.join(OK_DIR, filename))
                ok_count += 1
            else:
                label = "NG"
                shutil.move(filepath, os.path.join(NG_DIR, filename))
                ng_count += 1

            # 결과 저장
            results.append(
                {"filename": filename, "score": score, "classification": label}
            )

# CSV 저장
with open(CSV_FILE, "w", newline="") as csvfile:
    fieldnames = ["filename", "score", "classification"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)

# 통계 출력
print("? 분류 완료!")
print(f"정상(OK): {ok_count}개")
print(f"이상(NG): {ng_count}개")
print(f"결과 CSV 저장 위치: {CSV_FILE}")
