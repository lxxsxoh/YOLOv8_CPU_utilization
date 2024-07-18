from ultralytics import YOLO

# Load the exported TensorRT model
#tensorrt_model = YOLO("yolov8n-obb.engine", task='obb')
model = YOLO("yolov8n-obb.pt", task='obb')
model.to("cpu")


# Load the image path
image_path = "./dota_images"

#result = tensorrt_model(image_path)
result = model(image_path)

