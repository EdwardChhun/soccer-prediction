from ultralytics import YOLO

model = YOLO(r"soccer_detection3\weights\best.pt")

# Run inference on webcam
# https://docs.ultralytics.com/modes/predict/#inference-arguments
results = model.predict(
    source=r"assets\test.mp4", 
    show=True,
    conf = 0.4,
    line_width = 1)  # 0 for webcam

# Or for webcam
# results = model.predict(source="0", show=True)