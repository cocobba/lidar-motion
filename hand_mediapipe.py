import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 핸드 추정
    results = hands.process(image)

    # 이미지에 핸드 랜드마크 그리기
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 각 손가락의 랜드마크
            landmarks = hand_landmarks.landmark

            # 엄지 손가락 (Tip - IP joint)
            if landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x:
                finger_count += 1

            # 검지 손가락 (Tip - PIP joint)
            if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                finger_count += 1

            # 중지 손가락 (Tip - PIP joint)
            if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                finger_count += 1

            # 약지 손가락 (Tip - PIP joint)
            if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                finger_count += 1

            # 새끼 손가락 (Tip - PIP joint)
            if landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y:
                finger_count += 1

            # 손가락 개수 출력
            cv2.putText(image, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 결과 이미지 출력
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

