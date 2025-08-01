#برنامه تشخیص احساسات
import cv2
import face_recognition
from deepface import DeepFace


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not cap.isOpened():
    print("خطا: وب‌کم شناسایی نشد.")
    exit()


offset_y = 0

while True:
    success, frame = cap.read()
    if not success or frame is None or frame.size == 0:
        print("نمی‌توان فریم معتبر دریافت کرد.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        face_locations = face_recognition.face_locations(frame_rgb)
    except Exception as e:
        print(f"خطا در تشخیص چهره: {e}")
        continue

    for face_location in face_locations:
        top, right, bottom, left = face_location

        top = max(0, top + offset_y)
        bottom = min(frame.shape[0], bottom + offset_y)
        left = max(0, left)
        right = min(frame.shape[1], right)

        face_image = frame[top:bottom, left:right]

        emotion = "نامشخص"

        if face_image.size > 0:
            try:
                result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                print(f"خطا در تحلیل احساسات: {e}")
        else:
            print("چهره‌ی بریده‌شده نامعتبر است.")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Emotion Detection', frame)

    if cv2.waitKey(1) in [ord('q'), ord('e'), 27]:
        break

cap.release()
cv2.destroyAllWindows()