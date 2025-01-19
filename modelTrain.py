from ultralytics import YOLO
model=YOLO("yolov8n-cls.pt")
dir='.\COVID-19_Radiography_Dataset'
model.train(data=dir,epochs=1,imgsz=64)