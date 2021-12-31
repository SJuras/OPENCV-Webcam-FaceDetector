import cv2
from random import randrange

# detect the cascade, to use it later
trained_face_data = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')

# video capture (if you place the video file path, it will take in the video as an input)
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    # change frame to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(frameGray)

    # draw a rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(frame, "Person", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    # show frame
    cv2.imshow("Video Feed", frame)
    # pauses the termination of the program
    cv2.waitKey(1)


print("Code Execured")
