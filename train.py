from ultralytics import YOLO


model = YOLO('yolo11s.pt')


# Training.
results = model.train(
   data=r'D:\Developer\soccer-prediction\Human-Detection-3\data.yaml', # Path to your data.yaml file
   imgsz=640,
   epochs=20,
   batch=8,
   save=True,
   device = "cpu", # Change to CUDA if you have CUDA Toolkit
   pretrained = True,
   project = r'D:\Developer\soccer-prediction', # Path to your working directory
   name='soccer_prediction')