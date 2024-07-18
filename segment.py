from ultralytics import YOLO


# Load the exported TensorRT model
#tensorrt_model = YOLO("yolov8n-seg.engine", task='segment')
model = YOLO("yolov8n-seg.pt", task='segment')
model.to("cpu")


# Load the image path
image_path = "./coco_images"

#result = tensorrt_model(image_path)
result = model(image_path)
