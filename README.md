# Kidzcare-A-Real-Time-Child-Safety-Monitoring-System
Kidzcare is an intelligent real-time object detection system designed to prioritize child safety by leveraging advanced AI technologies. By combining the power of YOLOv8 for object detection and Supervision for processing and visualization, Kidzcare provides an efficient and scalable solution for monitoring children in predefined safety zones.

Key Features
1. Dataset Preparation
The project starts with the careful preparation of the dataset to ensure accurate detection. Annotations for the dataset were created using CVAT (Computer Vision Annotation Tool), enabling precise labeling of children and adults. These detailed annotations provide the foundation for effective model training, ensuring the system's high accuracy in distinguishing between object classes.

2. Object Detection with YOLOv8
Kidzcare employs YOLOv8, a state-of-the-art object detection model, to identify and differentiate children and adults in real-time video frames. The model's speed and precision ensure accurate detection, even in dynamic environments. This robust detection capability allows the system to monitor movement and maintain a count of individuals in the predefined area.

3. Advanced Filtering and Annotations
To refine the detection results, Supervision is integrated for filtering objects based on specific criteria, such as class IDs and confidence thresholds. Real-time annotations for bounding boxes and labels are added to video frames, providing a clear and interpretable visualization of the detected objects.

4. Safety Zone Monitoring
Using Supervision's polygon tools, Kidzcare defines safety zones within the monitored area. The system continuously tracks movement and identifies when children step outside these predefined boundaries, triggering alerts to ensure their safety.

5. Dynamic Video Processing
The system incorporates dynamic frame rate control to optimize video processing performance. By reducing the frame rate during low activity periods and increasing it during high motion activity, Kidzcare achieves efficient resource utilization without compromising accuracy.

Conclusion
Kidzcare offers an innovative and safety-driven monitoring solution by integrating powerful object detection with efficient video processing and zone-based tracking. Whether in schools, playgrounds, or public areas, this system ensures real-time child safety with precision and reliability.

Feel free to explore the code, dataset preparation process, and implementation details provided in this repository to learn more about the Kidzcare project! 🚀
