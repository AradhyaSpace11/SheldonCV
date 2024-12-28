from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change it if you have multiple cameras

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, save=False, conf=0.25)  # conf sets the confidence threshold

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Webcam Inference", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
