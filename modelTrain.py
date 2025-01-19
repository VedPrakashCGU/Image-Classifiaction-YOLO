from ultralytics import YOLO
model=YOLO("yolov8n-cls.pt")
dir='.\X-RayImageDataSet'
model.train(data=dir,epochs=10,imgsz=64)