import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Define paths
VIDEO_PATH = r"C:\Users\theekshana\Downloads\4182249-hd_720_1280_25fps.mp4"
MODEL_PATH = r"C:\Users\theekshana\Desktop\kidzcare weights-20240723T153013Z-001\kidzcare weights\Kidzcare adult kid detection models\94.1 63.3\best (4).pt"

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\theekshana\Desktop\FF\kidz-care-firebase-adminsdk-lz55k-8cffe6037c.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://kidz-care-default-rtdb.firebaseio.com/kids/count'
})

# Load YOLO model
model = YOLO(MODEL_PATH)

# Define polygon zone
def get_polygon_zone(image_shape):
    polygons = np.array([[168, 399], [176, 743], [620, 747], [600, 395]])
    height, width, _ = image_shape
    return sv.PolygonZone(polygon=polygons, frame_resolution_wh=(width, height))

# Annotators
def get_annotators(zone):
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, color=sv.Color.RED, thickness=5, text_thickness=10, text_scale=2
    )
    box_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    return zone_annotator, box_annotator

def update_firebase(interval_index, count):
    ref = db.reference('/kids/count')
    ref.set(count)
    print(f"Updated Firebase with interval {interval_index}: {count} detections")

def process_frame(frame: np.ndarray, frame_index: int, detection_counts: dict) -> np.ndarray:
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]

    # Count detections in the zone
    mask = zone.trigger(detections=detections)
    detections_filtered = detections[mask]
    
    # Count the number of detections
    count = len(detections_filtered)
    
    # Determine the current time in seconds
    current_time_sec = frame_index / fps

    # Record count at exact 5-second intervals
    if current_time_sec % 5 == 0:
        interval_index = int(current_time_sec // 5)
        detection_counts[interval_index] = count
        print(f"Time {int(current_time_sec)} seconds: {count} detections")

        # Update Firebase with the current count
        update_firebase(interval_index, count)

    # Annotate frame
    frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
    frame = zone_annotator.annotate(scene=frame)
    
    return frame

# Set up video capture and zone/annotators
video_capture = cv2.VideoCapture(VIDEO_PATH)
fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

zone = get_polygon_zone((frame_height, frame_width, 3))
zone_annotator, box_annotator = get_annotators(zone)

# Dictionary to store detection counts for exact 5-second intervals
detection_counts = {}

# Process video
def process_video_at_intervals():
    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = process_frame(frame, frame_index, detection_counts)
        frame_index += 1
    
    video_capture.release()

process_video_at_intervals()

# Optionally print detection counts at exact 5-second intervals
for interval_index in sorted(detection_counts.keys()):
    interval_start = interval_index * 5
    print(f"5-second interval {interval_start} seconds: {detection_counts[interval_index]} detections")
