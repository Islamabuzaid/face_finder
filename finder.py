import face_recognition 
import cv2

raed_img = face_recognition.load_image_file('img5.jpg')
brother_encoding = face_recognition.face_encodings(raed_img) #the 0 grabs the first face in the photo
# print(f"Found {len(brother_encoding)} faces in the photo") run to see if the picture is not the issue

frame_count = 0

webcam = cv2.VideoCapture(0) # use 1 if you have external webcam
while True:
    ret, frame = webcam.read()
    frame_count += 1
    face_locations = face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame, face_locations)

    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
        match = face_recognition.compare_faces([brother_encoding], face_encoding)[0]
        name = 'Brother' if match else 'Unknown'

        #draw box around face and labels it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()