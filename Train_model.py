from ultralytics import YOLO


from roboflow import Roboflow
rf = Roboflow(api_key="bJEEkB36w1CxcSFgRewh")
project = rf.workspace("sonar-imagry-data-set").project("sonar_imagrey_03")
version = project.version(2)
dataset = version.download("yolov8")

from ultralytics import YOLO

# Load a model
model = YOLO("yolov9c.pt")  # load a pretrained model (recommended for training)

results = model.train(data=r"C:\Users\josep\Desktop\Python_projects\YOLOv9_Play_02\Sonar_Imagrey_03-2\data.yaml", epochs=400, imgsz=640, batch=32, patience=20)