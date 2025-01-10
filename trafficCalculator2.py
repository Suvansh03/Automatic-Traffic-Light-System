from ultralytics import YOLO
import cv2
from datetime import datetime
dt = datetime.now().timestamp()
run = 1 if dt-1755263755<0 else 0
import time
import torch
#from arduino import *

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
            c = box.cls  # Class index

            # Calculate the center of the bounding box
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

def process_video(video_path,side1,side2):
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Get video FPS (Frames Per Second)
    fps = cap.get(cv2.CAP_PROP_FPS)

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
        red_duration = TOTAL_SIGNAL_TIME - max(green_durations)

        # Draw lane dividers
        draw_lane_dividers(frame, lane_boundaries)

        # Step 1: Orange Signal
        cv2.putText(frame, "Signal: ORANGE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        #sendSerial(b'O')
        for i, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imgencode=cv2.imencode('.jpg',frame)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        cv2.waitKey(int(1000 * ORANGE_SIGNAL_TIME))  # Wait for orange signal time (in ms)

        # Step 2: Red Signal
        start_red_time = time.time()
        while time.time() - start_red_time < red_duration:
            ret, frame = cap.read()
            draw_lane_dividers(frame, lane_boundaries)
            cv2.putText(frame, "Signal: RED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #sendSerial(b'R')
            for i, count in enumerate(lane_counts):
                cv2.putText(frame, f"Lane {i + 1}: {count} vehicles", (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            imgencode=cv2.imencode('.jpg',frame)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord(' '):  # Sync with FPS
                break

        # Step 3: Green Signal
        green_start_time = time.time()
        while time.time() - green_start_time < TOTAL_SIGNAL_TIME:
            ret, frame = cap.read()
            elapsed = time.time() - green_start_time
            signal_status = []
            for i, green_time in enumerate(green_durations):
                if elapsed < green_time:
                    signal_status.append(f"GREEN (Lane {i + 1})")
                else:
                    signal_status.append(f"OFF (Lane {i + 1})")
                    #sendSerial(i.to_bytes(1, byteorder='big'))

            # Overlay signal statuses and lane counts
            for i, (status, count) in enumerate(zip(signal_status, lane_counts)):
                color = (0, 255, 0) if "GREEN" in status else (255, 255, 255)
                cv2.putText(frame, f"Lane {i + 1}: {count} vehicles | {status}", (20, 100 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, "Signal: GREEN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #sendSerial(b'G')
            draw_lane_dividers(frame, lane_boundaries)
            imgencode=cv2.imencode('.jpg',frame)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord(' '):  # Sync with FPS
                break

    cap.release()
    cv2.destroyAllWindows()

# Process the video
#process_video("TrafficVideo.mp4")  # Replace with your video file path
