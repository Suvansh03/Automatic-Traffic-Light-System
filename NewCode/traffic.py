from ultralytics import YOLO
import cv2
import threading
import time
import torch

# Constants for traffic signal timing
TOTAL_SIGNAL_TIME = 10  # Total green + red time in seconds
ORANGE_SIGNAL_TIME = 2  # Fixed time for orange signal in seconds
MIN_GREEN_TIME = 1  # Minimum green time for any lane in seconds

# Global model
model = YOLO('yolov8n.pt')  # Replace with your trained YOLO model


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


def process_video(video_path, window_name, initial_signal, pos_x, pos_y):
    """
    Process video stream and alternate signals between two sides.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set smaller width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set smaller height

    # Get frame dimensions and calculate lane boundaries
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video {video_path}")
        return
    frame_height, frame_width, _ = frame.shape
    num_lanes = 3  # Number of lanes
    lane_boundaries = [int(i * frame_width / num_lanes) for i in range(num_lanes + 1)]  # Lane boundaries

    current_signal = initial_signal
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO prediction
        results = model.predict(frame, verbose=False)

        # Count vehicles and calculate green signal durations
        lane_counts = count_vehicles(frame, results, lane_boundaries)
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
            cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, signal_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, signal_color, 2)

        # Display frame in a separate window
        cv2.imshow(window_name, frame)
        cv2.moveWindow(window_name, pos_x, pos_y)

        # Alternate signals after TOTAL_SIGNAL_TIME
        if current_signal == "green":
            time.sleep(TOTAL_SIGNAL_TIME)
            current_signal = "red"
        else:
            time.sleep(TOTAL_SIGNAL_TIME)
            current_signal = "green"

        # Check for user interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


# Threaded video processing for two sides
video1_path = "TrafficVideo1.mp4"
video2_path = "TrafficVideo2.mp4"

def start_traffic_side_1():
    process_video(video1_path, "Side 1", "green", 100, 100)

def start_traffic_side_2():
    process_video(video2_path, "Side 2", "red", 500, 100)

# Create threads for both sides
thread1 = threading.Thread(target=start_traffic_side_1)
thread2 = threading.Thread(target=start_traffic_side_2)

# Start threads
thread1.start()
thread2.start()

# Wait for threads to complete
thread1.join()
thread2.join()

cv2.destroyAllWindows()
