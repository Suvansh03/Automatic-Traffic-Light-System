from ultralytics import YOLO
import cv2
import time
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755263755<0 else 0
import torch

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your trained model path

# Constants
TOTAL_SIGNAL_TIME = 30  # Total green signal time for both sides combined
ORANGE_SIGNAL_TIME = 3  # Duration for orange signal
MIN_GREEN_TIME = 5      # Minimum green time per lane
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
NUM_LANES = 3

def count_vehicles(results, lane_boundaries):
    """
    Count vehicles in each lane based on YOLO results.
    """
    lane_counts = [0] * (len(lane_boundaries) - 1)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].to(dtype=torch.float)
            center_x = (b[0] + b[2]) / 2
            for i in range(len(lane_boundaries) - 1):
                if lane_boundaries[i] <= center_x < lane_boundaries[i + 1]:
                    lane_counts[i] += 1
                    break
    return lane_counts

def calculate_lane_durations(lane_counts, total_green_time):
    """
    Allocate green time per lane based on vehicle count in each lane.
    """
    total_vehicles = sum(lane_counts)
    if total_vehicles == 0:
        return [MIN_GREEN_TIME] * len(lane_counts)
    return [max(MIN_GREEN_TIME, int((count / total_vehicles) * total_green_time)) for count in lane_counts]

def process_video(video_path, current_side, opposite_side):
    print('test')
    cap = cv2.VideoCapture(video_path)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    lane_boundaries = [int(i * FRAME_WIDTH / NUM_LANES) for i in range(NUM_LANES + 1)]

    while True:
        # Traffic counts for both sides
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        lane_counts = count_vehicles(results, lane_boundaries)
        lane_durations = calculate_lane_durations(lane_counts, TOTAL_SIGNAL_TIME)

        # Display lane counts
        for i, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (20, 50 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Signal transition
        for lane_idx, duration in enumerate(lane_durations):
            # Orange phase
            for _ in range(ORANGE_SIGNAL_TIME):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"Lane {lane_idx + 1}: ORANGE ({ORANGE_SIGNAL_TIME}s)", 
                            (20, 50 + lane_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                yield display_frame(frame)

            # Green phase
            green_start_time = time.time()
            while time.time() - green_start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"Lane {lane_idx + 1}: GREEN ({int(duration - (time.time() - green_start_time))}s)",
                            (20, 50 + lane_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                yield display_frame(frame)

            # Turn lane off (black)
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"Lane {lane_idx + 1}: OFF", (20, 50 + lane_idx * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                yield display_frame(frame)

        # Complementary signal: alternate between sides
        yield from handle_opposite_signal(opposite_side)

    cap.release()

def display_frame(frame):
    """
    Encode and yield the video frame.
    """
    draw_lane_dividers(frame)
    imgencode = cv2.imencode('.jpg', frame)[1]
    stringData = imgencode.tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + stringData + b'\r\n')

def draw_lane_dividers(frame):
    """
    Draw vertical lane divider lines on the frame.
    """
    for x in range(1, NUM_LANES):
        cv2.line(frame, (x * FRAME_WIDTH // NUM_LANES, 0), 
                 (x * FRAME_WIDTH // NUM_LANES, FRAME_HEIGHT), 
                 (0, 255, 0), 2)

def handle_opposite_signal(opposite_side):
    """
    Yield frames for the red signal phase of the current side, while
    the opposite side is green.
    """
    red_start_time = time.time()
    while time.time() - red_start_time < TOTAL_SIGNAL_TIME:
        frame = cv2.imread(f"{opposite_side}_frame.jpg")
        if frame is not None:
            cv2.putText(frame, f"{opposite_side.upper()}: GREEN", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            yield display_frame(frame)


