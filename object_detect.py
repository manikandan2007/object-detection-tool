import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time
engine = pyttsx3.init()
engine.setProperty('rate',150)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

model = YOLO("yolov8n.pt")
office_items = [
    "laptop",
    "mouse",
    "keyboard",
    "book",
    "bottle",
    "cup",
    "scissors",
    "cell phone"
]

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("External webcam not detected. Trying camera 0")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera started... Press Q to exit")


last_alert = {}
cooldown = 8

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received")
        break

    results = model(frame, conf=0.4)

    object_counts = {}
    current_time = time.time()

    for result in results:
        for box in result.boxes:

            label = model.names[int(box.cls[0])]

            if label in office_items:

                object_counts[label] = object_counts.get(label,0) + 1
                if label not in last_alert or current_time - last_alert[label] > cooldown:

                    print(label,"detected")

                    threading.Thread(
                        target=speak,
                        args=(f"{label} detected",),
                        daemon=True
                    ).start()

                    last_alert[label] = current_time

    if object_counts:

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open("object_log.txt","a") as file:
            file.write(f"{timestamp} -> {object_counts}\n")

    y = 30
    for obj,count in object_counts.items():

        cv2.putText(frame,
                    f"{obj}: {count}",
                    (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)

        y += 30

    annotated = results[0].plot()

    cv2.imshow("Office Detection System",annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()