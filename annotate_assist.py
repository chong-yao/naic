import cv2
import os
from ultralytics import YOLO
import numpy as np
import shutil

# Configuration
MODEL_PATH = "manually + assisted/kuih-lapis/kuih-lapis98m-seg.pt"

IMAGES_SOURCE = "[SCRAPED.png] kuih/kuih-lapis"

SAVE_DIR = "manually + assisted/kuih-lapis"

CONF_THRESH = 0.5

# Get screen dimensions
try:
    from win32api import GetSystemMetrics
    SCREEN_W = GetSystemMetrics(0)
    SCREEN_H = GetSystemMetrics(1)
    MAX_SCREEN_RATIO = 0.8  # Use 80% of screen space
except:
    # Fallback values if win32api not available
    SCREEN_W = 1920
    SCREEN_H = 1080
    MAX_SCREEN_RATIO = 0.8

# Create directories
os.makedirs(os.path.join(SAVE_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'labels'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'skipped'), exist_ok=True)

model = YOLO(MODEL_PATH).to('cuda')

def get_scaled_dimensions(img_w, img_h):
    # Calculate maximum allowed dimensions
    max_w = SCREEN_W * MAX_SCREEN_RATIO
    max_h = SCREEN_H * MAX_SCREEN_RATIO
    
    # Calculate scaling factors
    scale_w = max_w / img_w
    scale_h = max_h / img_h
    
    # Use the smaller scale factor to maintain aspect ratio
    scale = min(scale_w, scale_h)
    
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    return new_w, new_h

counter = 0

for img_name in os.listdir(IMAGES_SOURCE):


    counter+=1
    if not img_name.lower().endswith(('png',)):
        continue
        
    img_path = os.path.join(IMAGES_SOURCE, img_name)
    
    try:
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Skipping invalid image: {img_name}")
            continue
    except Exception as e:
        print(f"Error reading {img_name}: {str(e)}")
        continue

    display_image = original_image.copy()
    H, W = original_image.shape[:2]
    
    results = model.predict(original_image, conf=CONF_THRESH)
    label_lines = []

    for result in results:
        if result.masks is None or result.boxes is None:
            continue
        
        masks = result.masks.xy
        boxes = result.boxes
        
        if not len(masks) == len(boxes):
            continue
        
        for mask, box in zip(masks, boxes):
            try:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
            except IndexError:
                continue
            
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(display_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            x1, y1, _, _ = map(int, box.xyxy[0].tolist())
            cv2.putText(display_image, cls_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            normalized_mask = mask / np.array([W, H])
            flattened_mask = normalized_mask.reshape(-1).round(30).tolist()
            label_lines.append(f"{cls_id} {' '.join(map(str, flattened_mask))}")

    # Calculate scaled dimensions
    new_w, new_h = get_scaled_dimensions(W, H)
    
    # Resize the display image
    resized_image = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(counter)
    
    # Create resizable window
    cv2.namedWindow('Preview (A=Accept, S=Skip, ESC=Exit)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview (A=Accept, S=Skip, ESC=Exit)', new_w, new_h)
    cv2.imshow('Preview (A=Accept, S=Skip, ESC=Exit)', resized_image)
    
    key = cv2.waitKey(0)
    
    base_name = os.path.splitext(img_name)[0]
    
    if key == ord('a') and label_lines:
        try:
            dest_path = os.path.join(SAVE_DIR, 'images', img_name)
            shutil.move(img_path, dest_path)
            with open(os.path.join(SAVE_DIR, 'labels', f"{base_name}.txt"), 'w') as f:
                f.write('\n'.join(label_lines))
            print(f"Accepted: {img_name}")
        except Exception as e:
            print(f"Error saving {img_name}: {str(e)}")
    else:
        try:
            dest_path = os.path.join(SAVE_DIR, 'skipped', img_name)
            shutil.move(img_path, dest_path)
            print(f"Skipped: {img_name}")
        except Exception as e:
            print(f"Error moving {img_name}: {str(e)}")
        
        if key == 27:
            break

cv2.destroyAllWindows()