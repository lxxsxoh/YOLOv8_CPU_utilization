from ultralytics import YOLO


# Load the exported TensorRT model
#tensorrt_model = YOLO("yolov8n-cls.engine", task='classify')
model = YOLO("yolov8n-cls.pt", task='classify')
model.to("cpu")

# Load the image path
image_path = "./imagenet_images"


#result = tensorrt_model(image_path)
result = model(image_path)

