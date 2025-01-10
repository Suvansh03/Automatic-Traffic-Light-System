from ultralytics import YOLO
import cv2
import time
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


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    lane_boundaries = [int(i * FRAME_WIDTH / NUM_LANES) for i in range(NUM_LANES + 1)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        lane_counts = count_vehicles(results, lane_boundaries)
        lane_durations = calculate_lane_durations(lane_counts, TOTAL_SIGNAL_TIME)

        # Draw divider lines
        draw_lane_dividers(frame)

        # Orange phase
        for _ in range(ORANGE_SIGNAL_TIME):
            ret, frame = cap.read()
            if not ret:
                break
            draw_lane_dividers(frame)
            for i in range(NUM_LANES):
                cv2.putText(frame, f"Lane {i + 1}: ORANGE", (20, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.imshow("Traffic Signal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Red phase
        for _ in range(ORANGE_SIGNAL_TIME):  # Red time can be adjusted here
            ret, frame = cap.read()
            if not ret:
                break
            draw_lane_dividers(frame)
            for i in range(NUM_LANES):
                cv2.putText(frame, f"Lane {i + 1}: RED", (20, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Traffic Signal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # Green phase
        green_start_time = time.time()
        while time.time() - green_start_time < max(lane_durations):
            ret, frame = cap.read()
            if not ret:
                break
            draw_lane_dividers(frame)
            for i, duration in enumerate(lane_durations):
                remaining_time = max(0, duration - int(time.time() - green_start_time))
                color = (0, 255, 0) if remaining_time > 0 else (0, 0, 255)
                cv2.putText(frame, f"Lane {i + 1}: GREEN ({remaining_time}s)" if remaining_time > 0 else "RED",
                            (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Traffic Signal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    cap.release()
    cv2.destroyAllWindows()


def draw_lane_dividers(frame):
    """
    Draw equally spaced vertical lane divider lines on the frame.
    """
    for x in range(1, NUM_LANES):
        start_x = x * FRAME_WIDTH // NUM_LANES
        cv2.line(frame, (start_x, 0), (start_x, FRAME_HEIGHT), (255, 255, 255), 2)


if __name__ == "__main__":
    process_video("TrafficVideo.mp4")
