from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained YOLOv8 model
model = YOLO(r"D:\ROXs\SCS\25MLP4\Real_Time_Child_Safety_Monitoring_System\Model\best (4).pt")  
video_path = r"D:\ROXs\SCS\25MLP4\Real_Time_Child_Safety_Monitoring_System\Source\3195392-hd_1280_720_25fps.mp4" 
# video_path = r"D:\ROXs\SCS\25MLP4\Real_Time_Child_Safety_Monitoring_System\Source\3191109-hd_1366_720_25fps.mp4" 
cap = cv2.VideoCapture(video_path)

fps = 30  # Initial frame rate
low_fps = 15  # Lower frame rate when no motion detected
high_fps = 30  # Higher frame rate when motion detected

ret, prev_frame = cap.read()

def motion_detected(frame1, frame2, threshold=10000):
    # Calculate the absolute difference between frames
    diff = cv2.absdiff(frame1, frame2)
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
   
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh, None, iterations=3)
   
    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contour is detected, return True
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            return True
    return False

class_colors = {
    0: (255, 0, 0),    
    1: (0, 255, 0),   
}

while True:
    ret, img = cap.read()
    
    if not ret:
        break  # Exit the loop if there are no more frames

    # Perform inference on the captured frame
    results = model(img)  

    # Visualize and annotate the results on the frame
    for result in results:
        # Access detected boxes and show them
        detected_boxes = result.boxes  # Get the bounding boxes
        for box in detected_boxes:
            # Extract box coordinates and class ID
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get box coordinates
            class_id = int(box.cls[0])  # Get class label 

            color = class_colors.get(class_id, (255, 255, 255))  

            # Draw box with the selected color
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  
            cv2.putText(img, f'{box.conf[0]:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check for motion detection
    if motion_detected(prev_frame, img,400000):
        fps = high_fps  # Increase frame rate
        print("Harmful Motion detected. Increasing frame rate.")
    else:
        fps = low_fps  # Decrease frame rate
        print("No motion detected. Reducing frame rate.")

    cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', img)

    # Save the previous frame
    prev_frame = img

    # Exit the loop on 'q' key press
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
