import cv2
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array
from threading import Thread
import pygame
import time


def playSon(level):
    if level == 0:
        return
    else:
        if level == 1:
            audio = pygame.mixer.Sound('Sounds/1.wav')
        elif level == 2:
            audio = pygame.mixer.Sound('Sounds/2.wav')
        elif level == 3:
            audio = pygame.mixer.Sound('Sounds/3.wav')   
        audio.play()
        pygame.time.wait(500)
        pygame.mixer.stop()
    

def predict_left_eye(left_eye):
    global etat1
    for (x1, y1, w1, h1) in left_eye:
        eye1 = roi_gray[y1:y1+h1, x1:x1+w1]
        eye1 = cv2.GaussianBlur(eye1, (3, 3), 0.5)
        eye1 = cv2.resize(eye1, (img_size, img_size))
        eye1 = eye1.astype('float') / 255.0
        eye1 = img_to_array(eye1)
        eye1 = np.expand_dims(eye1, axis=0)
        pred1 = model.predict(eye1)
        etat1 = 1 if pred1 > 0.5 else 0
        break


def predict_right_eye(right_eye):
    global etat2
    for (x2, y2, w2, h2) in right_eye:
        eye2 = roi_gray[y2:y2 + h2, x2:x2 + w2]
        eye2 = cv2.GaussianBlur(eye2, (3, 3), 0.5)
        eye2 = cv2.resize(eye2, (img_size, img_size))
        eye2 = eye2.astype('float') / 255.0
        eye2 = img_to_array(eye2)
        eye2 = np.expand_dims(eye2, axis=0)
        pred2 = model.predict(eye2)
        etat2 = 1 if pred2 > 0.5 else 0
        break

def compute_drowsiness_score(blink_rate, perclos, blink_rate_threshold=15, perclos_threshold=25):
    if blink_rate < blink_rate_threshold and perclos > perclos_threshold:
        return 2
    elif blink_rate < blink_rate_threshold or perclos > perclos_threshold:
        return 1
    else:
        return 0





pygame.init()
pygame.mixer.init()
face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("assets/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("assets/haarcascade_righteye_2splits.xml")
model = load_model("model64/model.h5")
img_size = 64
etat1 = ''
etat2 = ''

score = ''

"""paramètres relatifs à l'approche"""
frame_count = 0
closed_count = 0
perclos = '_'
time_window = 60
start_time = time.time()

blink_count = 0

prev_open = True

blink_start_time = 0

closed_threshold = 0.5

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # application de filtre gaussien sur l'image pour la reduction de bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # chercher la plus grosse tete
    closest_face = None
    min_distance = float('inf')
    for (x, y, w, h) in faces:
        distance = abs(x - frame.shape[1] / 2) + abs(y - frame.shape[0] / 2)
        if distance < min_distance:
            closest_face = (x, y, w, h)
            min_distance = distance
    if closest_face is not None:
        frame_count += 1
        cv2.putText(frame, "score: " + str(score), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        roi_gray = gray[y:y+h, x:x+w]

        left_eye = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        left_eye_thread = Thread(target=predict_left_eye, args=(left_eye,))
        right_eye_thread = Thread(target=predict_right_eye, args=(right_eye,))

        left_eye_thread.start()
        right_eye_thread.start()

        left_eye_thread.join()
        right_eye_thread.join()

        if etat1 == 0 and etat2 == 0:
            closed_count += 1
            if prev_open:
                blink_count += 1
                blink_start_time = time.time()
                prev_open = False
            else:
                if time.time() - blink_start_time >= closed_threshold:
                    cv2.putText(frame, "closing time: " + str(time.time() - start_time) + "s", (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                    try:
                        playSon(3)
                    except:
                        continue

            cv2.putText(frame, "closed", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        else:
            prev_open = True
            cv2.putText(frame, "open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        if time.time() - start_time >= time_window:
            perclos = closed_count / frame_count * 100
            score = compute_drowsiness_score(blink_count, perclos)
            playSon(score)
            blink_count = 0
            closed_count = 0
            frame_count = 0
            start_time = time.time()

    cv2.imshow("detecteur", frame)
    cv2.moveWindow("detecteur", 350, 150)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()


