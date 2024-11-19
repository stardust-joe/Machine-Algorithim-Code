
import pathlib
from pathlib import Path
from ultralytics import YOLO
model=YOLO("1870i_132e.pt")

images=r"C:\Users\josep\Desktop\Python_projects\YOLOv9_Play_02\images"
image_list=list(Path(images).glob('**/*.png'))+list(Path(images).glob('**/*.jpg'))


results=model.predict(image_list, imgsz=640, conf=0.3, save=True, show=True)
#"2024-11-07_132105_Frame_2192.png"

