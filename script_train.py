from ultralytics import YOLO

# Carregar o modelo YOLO pré-treinado
model = YOLO('yolov8n.pt')

# Configurar o dataset e treinar
model.train(data='./data.yaml', epochs=300, imgsz=640, device = 0)
