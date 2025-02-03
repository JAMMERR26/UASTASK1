# UASTASK1

Fruit Detection Using YOLO



This code utilizes the YOLO (You Only Look Once) object detection model to detect and count unique fruits from two images: a front view and a back view. It ensures that fruits visible in both images are not counted twice by applying an Intersection over Union (IoU) based deduplication method.

1. Libraries used

ultralytics.YOLO: To load and utilize the YOLO model.
cv2 (OpenCV): For image reading and visualization.
numpy: For numerical operations (although not heavily used in this code).

2. Loading the YOLO Model and Images

YOLO Model: A pre-trained model ("best.pt") specifically trained to detect fruits is loaded.
Image Loading: Two images, "front_view.jpg" and "back_view.jpg", are loaded for detection.


3. Detecting Fruits in Images

Confidence Threshold (conf=0.6): Ensures only detections with confidence above 60% are considered valid.
IoU Threshold (iou=0.7): Used internally by YOLO to filter overlapping detections within the same image.

4. Extracting Detected Fruits

Bounding Box Coordinates: Each detected fruit is represented by its coordinates (x1, y1, x2, y2) which define the box around the fruit.
Class ID: Identifies the type of fruit detected.
Confidence Score: Indicates the model’s certainty about the detection.
Result Storage: Detections are stored in a list for further processing.

5. Calculating Intersection over Union (IoU)

Intersection Area: The overlapping region between two bounding boxes is calculated.
Union Area: The total area covered by both bounding boxes, minus the intersection area.
IoU Value: A ratio representing how much two boxes overlap (ranges from 0 to 1). An IoU of 1 means perfect overlap, while 0 means no overlap.

7. Identifying Unique Fruits

Duplicate Check: Each fruit detected in the back image is compared with those from the front image.
Class Matching: Only fruits of the same type (same class ID) are compared.
IoU Threshold for Duplicates: If the IoU value exceeds 0.5, the fruit is considered a duplicate.
Unique Fruit List: Non-duplicate fruits from the back image are added to the list of unique fruits.

7. Counting and Displaying Results

Counting Unique Fruits: The total number of unique fruits is determined by counting the items in the final list.
Displaying Detections: The detected fruits are visualized with bounding boxes drawn on the original images.
Key Parameters to Adjust
Confidence Threshold (conf=0.6): Increase to reduce false positives; decrease to capture more objects.
IoU Threshold (iou_threshold=0.5): Adjust to control duplicate detection sensitivity. Lower values may count duplicates as unique.

