from ultralytics import YOLO
import cv2
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755263755<0 else 0
import time
import torch



# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your trained model path

# Constants
TOTAL_SIGNAL_TIME = 10  # Total time for both signals combined
ORANGE_SIGNAL_TIME = 2  # Fixed time for orange signal
MIN_GREEN_TIME = 1  # Minimum green time for any lane
shared_traffic_counts = {'side1': [], 'side2': []}  # Shared traffic counts

def draw_lane_dividers(frame, lane_boundaries):
    """
    Draw vertical lane divider lines on the frame.
    """
    height, width, _ = frame.shape
    for x in lane_boundaries[1:]:
        cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 2)  # Green vertical lines

def count_vehicles(frame, results, lane_boundaries):
    """
    Count vehicles in each lane based on YOLO results.
    """
    lane_counts = [0] * (len(lane_boundaries) - 1)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].to(dtype=torch.float)
            c = box.cls
            center_x = (b[0] + b[2]) / 2
            for i in range(len(lane_boundaries) - 1):
                if lane_boundaries[i] <= center_x < lane_boundaries[i + 1]:
                    lane_counts[i] += 1
                    break
    return lane_counts

def calculate_signal_durations(side1_counts, side2_counts):
    """
    Calculate green signal durations for both sides based on traffic.
    """
    total_vehicles = sum(side1_counts) + sum(side2_counts)
    if total_vehicles == 0:
        return [MIN_GREEN_TIME] * len(side1_counts), [MIN_GREEN_TIME] * len(side2_counts)
    
    side1_total = sum(side1_counts)
    side2_total = sum(side2_counts)

    side1_green_duration = max(MIN_GREEN_TIME, int((side1_total / total_vehicles) * (TOTAL_SIGNAL_TIME - ORANGE_SIGNAL_TIME)))
    side2_green_duration = TOTAL_SIGNAL_TIME - ORANGE_SIGNAL_TIME - side1_green_duration

    return [side1_green_duration] * len(side1_counts), [side2_green_duration] * len(side2_counts)

def process_video(video_path, side_name,side2):
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)
    cap.set(4, 480)
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    num_lanes = 3
    lane_boundaries = [int(i * frame_width / num_lanes) for i in range(num_lanes + 1)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction and lane count
        results = model.predict(frame, verbose=False)
        lane_counts = count_vehicles(frame, results, lane_boundaries)
        shared_traffic_counts[side_name] = lane_counts

        # Synchronize green signal durations
        side1_durations, side2_durations = calculate_signal_durations(shared_traffic_counts['side1'], shared_traffic_counts['side2'])
        green_durations = side1_durations if side_name == 'side1' else side2_durations
        red_duration = TOTAL_SIGNAL_TIME - max(green_durations)

        # Draw lane dividers
        draw_lane_dividers(frame, lane_boundaries)

        # Display signal statuses
        cv2.putText(frame, "Signal: ORANGE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        for i, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

        # Red and green signal phases
        phases = [("RED", red_duration), ("GREEN", green_durations[0])]
        for phase, duration in phases:
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                draw_lane_dividers(frame, lane_boundaries)
                for i, count in enumerate(lane_counts):
                    status_color = (0, 255, 0) if phase == "GREEN" else (0, 0, 255)
                    cv2.putText(frame, f"Lane {i + 1}: {count} vehicles | {phase} ({duration}s)",
                                (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                imgencode = cv2.imencode('.jpg', frame)[1]
                stringData = imgencode.tostring()
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
    cap.release()


