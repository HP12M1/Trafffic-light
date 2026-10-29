import cv2
import time
from ultralytics import YOLO

# ==========================================
# SMART TRAFFIC MANAGEMENT SYSTEM
# Using Image Processing and Computer Vision
# Four-Way Intersection
# ==========================================

# 1. Load YOLO model
print("Loading vehicle detection model...")
model = YOLO("yolov8n.pt")

# 2. Open USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not found!")
    print("Please connect your USB camera.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

# 3. Vehicle classes in the YOLO COCO model
# 2 = Car, 3 = Motorcycle, 5 = Bus, 7 = Truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# 4. Define the four traffic regions (x1, y1, x2, y2)
REGIONS = {
    "North": (180, 0, 460, 160),
    "South": (180, 320, 460, 480),
    "East":  (420, 130, 640, 350),
    "West":  (0, 130, 220, 350)
}

# 5. Traffic timing settings
SECONDS_PER_CAR = 2.5
MAX_CARS = 4
MAX_GREEN_TIME = SECONDS_PER_CAR * MAX_CARS
YELLOW_TIME = 2.0
ALL_RED_TIME = 1.0
MIN_GREEN_TIME = 2.5

# 6. Initial traffic signal state
current_direction = None
signal_state = "ALL_RED"
state_start_time = time.time()
green_duration = 0.0

# ==========================================
# FUNCTIONS
# ==========================================
def calculate_green_time(vehicle_count):
    """
    Calculates green time based on vehicle count.
    Applies MIN_GREEN_TIME and MAX_GREEN_TIME bounds.
    """
    if vehicle_count == 0:
        return 0.0
    limited_count = min(vehicle_count, MAX_CARS)
    calculated_time = limited_count * SECONDS_PER_CAR
    return max(calculated_time, MIN_GREEN_TIME)

def count_vehicles(result):
    counts = {direction: 0 for direction in REGIONS}
    if result.boxes is None:
        return counts

    for detection in result.boxes:
        class_id = int(detection.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        for direction, (rx1, ry1, rx2, ry2) in REGIONS.items():
            if rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2:
                counts[direction] += 1
                break

    return counts

def select_next_direction(counts, previous_direction):
    """
    Selects direction with the highest vehicle count.
    In case of a tie, maintains the current direction if eligible.
    """
    maximum_count = max(counts.values())
    if maximum_count == 0:
        return None

    candidates = [
        direction for direction, count in counts.items()
        if count == maximum_count
    ]

    # Keep previous direction during a tie
    if previous_direction in candidates:
        return previous_direction

    return candidates[0]

def draw_regions(frame, counts):
    for direction, (x1, y1, x2, y2) in REGIONS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        label = f"{direction}: {counts[direction]} vehicles"
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(y1 + 25, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

# ==========================================
# MAIN PROGRAM
# ==========================================
print("\nSmart Traffic Management System Started")
print("Press Q to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read camera frame.")
            break

        # Run YOLO vehicle detection
        result = model(frame, conf=0.35, verbose=False)[0]

        # Count vehicles & calculate green time
        vehicle_counts = count_vehicles(result)
        green_times = {
            direction: calculate_green_time(count)
            for direction, count in vehicle_counts.items()
        }

        busiest_direction = select_next_direction(vehicle_counts, current_direction)

        now = time.time()
        elapsed = now - state_start_time

        # ------------------------------
        # TRAFFIC SIGNAL STATE MACHINE
        # ------------------------------
        if signal_state == "ALL_RED":
            if elapsed >= ALL_RED_TIME:
                if busiest_direction is None:
                    current_direction = None
                    green_duration = 0.0
                    state_start_time = now
                else:
                    current_direction = busiest_direction
                    green_duration = green_times[current_direction]
                    if green_duration > 0:
                        signal_state = "GREEN"
                    state_start_time = now

        elif signal_state == "GREEN":
            if elapsed >= green_duration:
                signal_state = "YELLOW"
                state_start_time = now

        elif signal_state == "YELLOW":
            if elapsed >= YELLOW_TIME:
                signal_state = "ALL_RED"
                state_start_time = now

        # ------------------------------
        # HUD / DISPLAY RENDER
        # ------------------------------
        display = result.plot()
        draw_regions(display, vehicle_counts)

        # Status Overlay
        color = (0, 255, 0) if signal_state == "GREEN" else (0, 255, 255) if signal_state == "YELLOW" else (0, 0, 255)
        cv2.putText(display, f"Signal: {signal_state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        dir_text = current_direction if current_direction is not None else "None"
        cv2.putText(display, f"Direction: {dir_text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"Green time: {green_duration:.1f} sec", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if signal_state == "GREEN":
            remaining = max(0.0, green_duration - elapsed)
            cv2.putText(display, f"Remaining: {remaining:.1f} sec", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        y_pos = 150
        for direction in REGIONS:
            text = f"{direction}: {vehicle_counts[direction]} vehicles, {green_times[direction]:.1f} sec"
            cv2.putText(display, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25

        busiest_text = busiest_direction if busiest_direction is not None else "No vehicles"
        cv2.putText(display, f"Highest traffic: {busiest_text}", (10, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Smart Traffic Management System", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nSmart Traffic Management System Stopped.")
