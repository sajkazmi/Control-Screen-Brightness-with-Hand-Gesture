import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import screen_brightness_control as sbc

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,     
    min_detection_confidence=0.5 
)

cam = cv.VideoCapture(0)

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    index_palm_distance = ((index_tip.x - palm_base.x) ** 2 + (index_tip.y - palm_base.y) ** 2) ** 0.5
    middle_palm_distance = ((middle_tip.x - palm_base.x) ** 2 + (middle_tip.y - palm_base.y) ** 2) ** 0.5
    ring_palm_distance = ((ring_tip.x - palm_base.x) ** 2 + (ring_tip.y - palm_base.y) ** 2) ** 0.5
    pinky_palm_distance = ((pinky_tip.x - palm_base.x) ** 2 + (pinky_tip.y - palm_base.y) ** 2) ** 0.5

    threshold = 0.4 

    if (thumb_index_distance < threshold and
        index_palm_distance < threshold and
        middle_palm_distance < threshold and
        ring_palm_distance < threshold and
        pinky_palm_distance < threshold):
        return True
    return False

while cam.isOpened():
    success, frame = cam.read()

    if not success:
        print("Camera Frame not available")
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    hands_detected = hands.process(frame)

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
            if is_fist(hand_landmarks):
                sbc.set_brightness(20)  
            else:
                sbc.set_brightness(100)  

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
