from ultralytics import YOLO

model_zebra = YOLO("models/best.pt")
model_light = YOLO("models/traffic_light.pt")

# Get metrics stored inside the model from training
print("=== ZEBRA MODEL ===")
print("Training metrics:", model_zebra.ckpt.get('train_results', 'not found'))
print("Model args:", model_zebra.ckpt.get('train_args', 'not found'))

print("\n=== TRAFFIC LIGHT MODEL ===")
print("Training metrics:", model_light.ckpt.get('train_results', 'not found'))