from ultralytics import YOLO

# Load the model
model = YOLO(r"soccer_detection3\weights\best.pt")

# Perform inference
results = model(r"assets\low_res.jpeg")

# Process results
for r in results:
    print(r.boxes)  # print the detection boxes
    r.show()        # display the image with bounding boxes