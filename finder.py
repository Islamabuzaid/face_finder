import PIL.Image
import PIL.ImageOps
import numpy as np
import face_recognition
import cv2
from collections import deque

img = PIL.Image.open('img5.jpg').convert('RGB') # converts img to rgb format
img = PIL.ImageOps.exif_transpose(img)  # fixes phone camera rotation
raed_img = np.array(img)

encodings = face_recognition.face_encodings(raed_img)
print(f"Found {len(encodings)} faces")
brother_encoding = encodings[0] # grabs the first face found in the photo

frame_count = 0 
face_encoding = []
face_locations = []
stable_match = False

match_history = deque(maxlen=15) # keeps history of last 5 match results

webcam = cv2.VideoCapture(0) # use 1 if you have external webcam
while True:
    ret, frame = webcam.read()
    frame_count += 1
    if frame_count % 2 == 0: # analyzes every other fram (to make it fasterish)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_locations = face_recognition.face_locations(small_frame, model='hog') # hog model is faster but less accurate then default
        face_encoding = face_recognition.face_encodings(small_frame, small_locations)
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in small_locations]


    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
        match = face_recognition.compare_faces([brother_encoding], face_encoding, tolerance = 0.5)[0]
        match_history.append(match)

        confidence = match_history.count(True) / len(match_history) # only label if most of the frames agree
        
        if confidence >= 0.50:
            stable_match = True
        elif confidence < 0.50:
            stable_match = False


        name = 'Freaky ash' if stable_match else 'Unknown'

        #draw box around face and labels it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()