from ultralytics import YOLO


model = YOLO('yolo11s.pt')


# Training.
results = model.train(
   data=r'D:\Developer\soccer-prediction\Human-Detection-3\data.yaml',
   imgsz=640,
   epochs=20,
   batch=8,
   save=True,
   device = "cpu",
   pretrained = True,
   name='soccer_detection')