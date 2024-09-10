import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 포즈 추정
    results = pose.process(image)

    # 이미지에 포즈 랜드마크 그리기
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 추출
        landmarks = results.pose_landmarks.landmark

        # 왼쪽 팔(어깨, 팔꿈치, 손목) 랜드마크 추출
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        # 왼쪽 팔이 위로 올라갔는지 감지 (어깨보다 팔꿈치와 손목이 위에 있는 경우)
        if left_elbow.y < left_shoulder.y and left_wrist.y < left_elbow.y:
            cv2.putText(image, 'Left arm raised!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 오른쪽 팔도 추가 가능
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        if right_elbow.y < right_shoulder.y and right_wrist.y < right_elbow.y:
            cv2.putText(image, 'Right arm raised!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 이미지 출력
    cv2.imshow('Pose Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

