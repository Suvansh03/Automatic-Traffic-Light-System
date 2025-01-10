from ultralytics import YOLO
import cv2
from datetime import datetime
import time
import torch

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your trained model path

# Traffic signal constants
TOTAL_SIGNAL_TIME = 10  # Total time for green + red signals (in seconds)
ORANGE_SIGNAL_TIME = 2  # Fixed time for orange signal (in seconds)
MIN_GREEN_TIME = 1  # Minimum green time for any lane (in seconds)


def draw_lane_dividers(frame, lane_boundaries):
    """
    Draw vertical lane divider lines on the frame.
    """
    height, width, _ = frame.shape
    for x in lane_boundaries[1:]:  # Exclude the first boundary (0)
        cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 2)  # Green vertical lines


def count_vehicles(frame, results, lane_boundaries):
    """
    Count vehicles in each lane based on YOLO results.
    """
    lane_counts = [0] * (len(lane_boundaries) - 1)  # Initialize counts for each lane
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].to(dtype=torch.float)  # Bounding box coordinates
            center_x = (b[0] + b[2]) / 2  # Average of left and right X-coordinates

            # Assign to a lane based on center_x
            for i in range(len(lane_boundaries) - 1):
                if lane_boundaries[i] <= center_x < lane_boundaries[i + 1]:
                    lane_counts[i] += 1
                    break
    return lane_counts


def calculate_signal_durations(lane_counts):
    """
    Calculate green signal durations for each lane.
    """
    total_vehicles = sum(lane_counts)
    green_durations = [
        max(MIN_GREEN_TIME, int((count / total_vehicles) * (TOTAL_SIGNAL_TIME - ORANGE_SIGNAL_TIME))) if total_vehicles > 0 else MIN_GREEN_TIME
        for count in lane_counts
    ]
    return green_durations


def process_video(video_path, current_signal):
    """
    Process video stream and alternate signals between two sides.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Get frame dimensions
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    num_lanes = 3  # Number of lanes
    lane_boundaries = [int(i * frame_width / num_lanes) for i in range(num_lanes + 1)]  # Lane boundaries

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO prediction and count vehicles
        results = model.predict(frame, verbose=False)
        lane_counts = count_vehicles(frame, results, lane_boundaries)

        # Calculate green signal durations
        green_durations = calculate_signal_durations(lane_counts)

        # Signal Logic
        if current_signal == "red":
            signal_color = (0, 0, 255)  # Red signal
            signal_text = "Signal: RED"
        else:
            signal_color = (0, 255, 0)  # Green signal
            signal_text = "Signal: GREEN"

        # Draw lane dividers
        draw_lane_dividers(frame, lane_boundaries)

        # Display signal status and vehicle counts
        for i, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, signal_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)

        # Encode and stream the frame
        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stringData + b'\r\n')

        # Alternate signals after TOTAL_SIGNAL_TIME
        time.sleep(TOTAL_SIGNAL_TIME)
        current_signal = "green" if current_signal == "red" else "red"

    cap.release()
    cv2.destroyAllWindows()
