import cv2
import mediapipe as mp
import pyautogui
import math

# PyAutoGUI Configuration
pyautogui.FAILSAFE = False  # Prevents crash if cursor hits screen corners
pyautogui.PAUSE = 0         # Removes internal delays for smooth cursor motion

# Screen Size
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam Setup
cap = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480
cap.set(3, cam_w)
cap.set(4, cam_h)

# Motion & Threshold Variables
frame_margin = 100       # Boundary box margin in pixels
smoothing_factor = 5    # Higher = smoother cursor, but slight latency
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Click State
is_clicked = False
PINCH_THRESHOLD = 30     # Distance in pixels between Thumb and Index tip to register click

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert color space BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw Active Interaction Boundary Box
    cv2.rectangle(frame, (frame_margin, frame_margin), 
                  (cam_w - frame_margin, cam_h - frame_margin), 
                  (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Extract coordinates for Thumb Tip (4) and Index Tip (8)
            thumb_x = int(landmarks[4].x * cam_w)
            thumb_y = int(landmarks[4].y * cam_h)
            index_x = int(landmarks[8].x * cam_w)
            index_y = int(landmarks[8].y * cam_h)

            # Draw indicators on fingertips
            cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)

            # -------------------------------------------------------------
            # 1. CURSOR MOVEMENT (Map Camera ROI to Screen Resolution)
            # -------------------------------------------------------------
            # Clamp index coordinates within active boundary box
            clamped_x = max(frame_margin, min(index_x, cam_w - frame_margin))
            clamped_y = max(frame_margin, min(index_y, cam_h - frame_margin))

            # Interpolate coordinates to full desktop screen dimensions
            target_x = int(((clamped_x - frame_margin) / (cam_w - 2 * frame_margin)) * screen_w)
            target_y = int(((clamped_y - frame_margin) / (cam_h - 2 * frame_margin)) * screen_h)

            # Smooth cursor position using Exponential Moving Average
            curr_x = prev_x + (target_x - prev_x) / smoothing_factor
            curr_y = prev_y + (target_y - prev_y) / smoothing_factor

            # Move System Mouse Pointer
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # -------------------------------------------------------------
            # 2. PINCH GESTURE DETECT (Index to Thumb Distance)
            # -------------------------------------------------------------
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Line connecting Thumb and Index
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 255), 2)

            if distance < PINCH_THRESHOLD:
                # Highlight connection point in RED on pinch
                mid_x, mid_y = (index_x + thumb_x) // 2, (index_y + thumb_y) // 2
                cv2.circle(frame, (mid_x, mid_y), 12, (0, 0, 255), -1)

                if not is_clicked:
                    pyautogui.click()
                    is_clicked = True
                    cv2.putText(frame, "CLICK!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                is_clicked = False

    # Status Overlay
    cv2.putText(frame, "Virtual Mouse Active", (20, cam_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Virtual Mouse - Pinch Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()