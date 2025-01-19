from ultralytics import YOLO
import numpy as np
model=YOLO(r'C:\Users\user\Desktop\New folder\runs\classify\train5\weights\best.pt')
img='.\pnemonia1.jpeg'
result=model(img)
names_dir=result[0].names
probs=result[0].probs.data.tolist()
print(names_dir[np.argmax(probs)])