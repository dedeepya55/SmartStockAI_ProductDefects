from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

image_path = "dataset/images/p1.JPG"
results = model(image_path)

img = cv2.imread(image_path)
damaged_found = False

for r in results:
    for box in r.boxes:
        damaged_found = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max(x2 - x1, y2 - y1) // 2

        cv2.circle(img, (cx, cy), radius, (0, 0, 255), 3)

cv2.imwrite("output.jpg", img)

if damaged_found:
    print("❌ Damaged product detected")
else:
    print("✅ Product is OK")
