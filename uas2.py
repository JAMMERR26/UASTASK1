from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("best.pt")  

front_image = cv2.imread("front_view.jpg")
back_image = cv2.imread("back_view.jpg")

if front_image is None or back_image is None:
    raise ValueError("UPLOAD THE IMAGE ")


front_results = model(front_image)
back_results = model(back_image)


def GETFRUITS(results):
    fruits = []
    for result in results:
        boxes = result.boxes  
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])  
            classid = int(boxes.cls[i].item()) 
            if classid == 3:
                continue
            conf = boxes.conf[i].item()  
            fruits.append(((x1, y1, x2, y2), classid, conf))
    return fruits



front_fruits = GETFRUITS(front_results)
back_fruits = GETFRUITS(back_results)


def iou_(box1, box2):
    x1A, y1A, x2A, y2A = box1
    x1B, y1B, x2B, y2B = box2

    
    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

   
    boxA_area = (x2A - x1A) * (y2A - y1A)
    boxB_area = (x2B - x1B) * (y2B - y1B)
    union_area = boxA_area + boxB_area - inter_area

    
    return inter_area / union_area if union_area > 0 else 0


iou_threshold = 0.5  
unique_fruits = front_fruits.copy()

for back_fruit in back_fruits:
    x1_b, y1_b, x2_b, y2_b = back_fruit[0]
    class_b = back_fruit[1]
    if class_b ==3:
        continue

    is_duplicate = False
    for front_fruit in front_fruits:
        x1_f, y1_f, x2_f, y2_f = front_fruit[0]
        class_f = front_fruit[1]

        if class_b == class_f: 
            iou = iou_((x1_f, y1_f, x2_f, y2_f), (x1_b, y1_b, x2_b, y2_b))
            if iou > iou_:
                is_duplicate = True
                break

    if not is_duplicate:
        unique_fruits.append(back_fruit)


total_fruit_count = len(unique_fruits)
print(f"TOTAL FRUITS : {total_fruit_count}")



for result in front_results:
    result.show()  # Show detections on front image
for result in back_results:
    result.show()  # Show detections on back image



