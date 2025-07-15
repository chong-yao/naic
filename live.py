# Load CNN model
from ultralytics import YOLO
import timm
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter
import numpy as np


cnn = YOLO("models/cnn.pt")

# Load ViT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MODIFIED: Create the model and move it to the defined device
vit = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=8).to(device)

# MODIFIED: Load the model weights, mapping them to the same device
vit.load_state_dict(torch.load("models/vit.pth", map_location=device))

# MODIFIED: Set to evaluation mode (no .cpu() needed)
vit.eval()
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load labels

with open('models/labels.txt') as f:
    labels = {i: line.strip().split(' ', 1)[1] for i, line in enumerate(f)}

print(labels)

num_classes = 8

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # --- 1. CNN Prediction (on GPU) ---
    # YOLO automatically uses the GPU if available
    cnn_results = cnn(frame, verbose=False)
    cnn_pred_classes = cnn_results[0].boxes.cls.tolist()

    if len(cnn_pred_classes) > 0:
        cnn_counts = Counter(cnn_pred_classes)
        cnn_probs = [cnn_counts.get(i, 0) for i in range(num_classes)]
        cnn_probs = [p / sum(cnn_probs) for p in cnn_probs]
        cnn_predicted_class_idx = int(max(cnn_counts, key=cnn_counts.get))
    else:
        cnn_probs = [0.0] * num_classes
        cnn_predicted_class_idx = -1

    # --- 2. ViT Prediction (on GPU) ---
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # MODIFIED: Move the input tensor to the same device as the model
    input_tensor = vit_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        vit_logits = vit(input_tensor)
    
    # Probabilities can be moved to CPU for subsequent logic
    vit_probs_tensor = F.softmax(vit_logits, dim=1).squeeze(0).cpu()
    vit_probs = vit_probs_tensor.tolist()
    vit_predicted_class_idx = int(torch.argmax(vit_probs_tensor).item())

    # --- 3. Combine Predictions (Ensemble) ---
    final_probs = [(c + v) / 2 for c, v in zip(cnn_probs, vit_probs)]

    if cnn_predicted_class_idx == vit_predicted_class_idx and cnn_predicted_class_idx != -1:
        final_idx = cnn_predicted_class_idx
    elif cnn_predicted_class_idx == -1 and vit_predicted_class_idx != -1:
        final_idx = vit_predicted_class_idx
    elif vit_predicted_class_idx == -1 and cnn_predicted_class_idx != -1:
        final_idx = cnn_predicted_class_idx
    elif cnn_predicted_class_idx != -1 and vit_predicted_class_idx != -1:
        final_idx = int(torch.tensor(final_probs).argmax().item())
    else:
        final_idx = -1

    final_label = labels.get(final_idx, 'Unknown')

    # --- 4. Display the result on the frame ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Prediction: {final_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam Live Inference (CUDA)', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()