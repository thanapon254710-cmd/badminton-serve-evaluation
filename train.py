from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="shuttle_dataset/data.yaml",
    epochs=100,
    imgsz=960,
    batch=16,
    patience=20,
    name="shuttlecock_v1"
)