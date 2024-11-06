from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference with bounding boxes displayed on the image
results = model.predict(source="images/bus.jpg", conf=0.4)

# Retrieve the annotated image directly from the results
for result in results:
    annotated_img = result.plot()  # This provides the image with bounding boxes drawn

    # Display the annotated image with OpenCV
    cv2.imshow("Inference Result", annotated_img)
    cv2.waitKey(0)  # Keep the window open until a key is pressed
    cv2.destroyAllWindows()
