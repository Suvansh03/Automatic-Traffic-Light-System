import cv2
import torch
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755263755<0 else 0
from ultralytics import YOLO

# Load YOLO model globally for better performance
model = YOLO("yolov8n.pt")  # Update with the path to your YOLOv8 model

def draw_lane_dividers(frame, lane_boundaries):
    """
    Draws lane dividers on the frame based on lane boundaries.
    """
    frame_height = frame.shape[0]
    for boundary in lane_boundaries:
        cv2.line(frame, (boundary, 0), (boundary, frame_height), (255, 255, 255), 2)
    return frame

def calculate_green_signals(lane_counts, max_green_time):
    """
    Calculates green signal duration for each lane based on vehicle counts.
    """
    return [min(max_green_time, count * 2) for count in lane_counts]

def detect_vehicles(frame, lane_boundaries):
    """
    Uses YOLO to detect vehicles in the frame and count them per lane.
    """
    results = model(frame)
    vehicle_classes = [2, 3, 5, 7]  # Example: Cars, motorbikes, buses, and trucks
    lane_counts = [0] * (len(lane_boundaries) - 1)

    for result in results:
        # Access the bounding boxes and their associated classes
        for box in result.boxes:
            cls = int(box.cls)  # Get the class index
            if cls in vehicle_classes:
                # Access bounding box coordinates in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
                x_center = (x1 + x2) / 2  # Calculate the center x-coordinate

                # Determine the lane where the vehicle is located
                for i in range(len(lane_boundaries) - 1):
                    if lane_boundaries[i] <= x_center < lane_boundaries[i + 1]:
                        lane_counts[i] += 1
                        break
    return lane_counts


def process_video(video_path, side_signal=10, num_lanes=3, max_green_time=10):
    """
    Processes a video, handling lane dividers, green signals, and red signals.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Get video FPS (Frames Per Second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine frame dimensions and calculate lane boundaries
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        return
    frame_height, frame_width, _ = frame.shape
    lane_boundaries = [int(i * frame_width / num_lanes) for i in range(num_lanes + 1)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles and calculate green durations
        lane_counts = detect_vehicles(frame, lane_boundaries)
        green_durations = calculate_green_signals(lane_counts, max_green_time)

        # Draw lane dividers
        frame = draw_lane_dividers(frame, lane_boundaries)

        # Side A: Green Signal
        elapsed = 0
        while elapsed < max_green_time:
            elapsed += 1 / fps
            for i, green_time in enumerate(green_durations):
                remaining_time = max(0, int(green_time - elapsed))
                status = f"Lane {i + 1}: {lane_counts[i]} vehicles | GREEN - {remaining_time}s"
                color = (0, 255, 0) if remaining_time > 0 else (255, 255, 255)
                cv2.putText(frame, status, (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            imgencode = cv2.imencode('.jpg', frame)[1]
            stringData = imgencode.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stringData + b'\r\n')

        # Side B: Red Signal (Opposite Green)
        elapsed = 0
        while elapsed < side_signal:
            elapsed += 1 / fps
            cv2.putText(frame, "Signal: RED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # imgencode = cv2.imencode('.jpg', frame)[1]
            # stringData = imgencode.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + stringData + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

def process_video1(video_path, side_signal=10, num_lanes=3, max_green_time=10):
    """
    Separate processing function for the second video stream.
    """
    return process_video(video_path, side_signal, num_lanes, max_green_time)
