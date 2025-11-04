from roboflow import Roboflow
rf = Roboflow(api_key="fnhoKzjnwnobE7LM9J9J")
project = rf.workspace("sarann").project("food-detect-qdnvk")
version = project.version(7)
dataset = version.download("yolov8")