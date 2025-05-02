from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")
results = model.train(
    data="apple.yaml",
    epochs=100,
    imgsz=256,
    batch = 4,
    project="apple_yolo",
    name="apple",
    optimizer="auto",
    cache=False,
    workers=8,
    lr0=0.01,
    lrf=0.01,
    dropout=0.2,
    save=True,
    plots=True,
    seed=42,
    half=True,
    device="cuda"
    )
