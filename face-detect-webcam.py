import cv2
import time
from keras.models import load_model
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier('./assets/haarcascade_frontalface_alt.xml')

vs = cv2.VideoCapture(0)

model = load_model('./assets/trained_model')

IMG_SIZE = 300


def evaluate_img(input_model, img, input_frame, pos_x, pos_y):
    img = Image.fromarray(img, 'RGB')
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = np.array(img)
    result = np.array([img.reshape(IMG_SIZE, IMG_SIZE, 1)])
    result = input_model.predict(result, verbose=0)
    if result[0][0] > 0.5:
        cv2.putText(img=input_frame, text="Chinese", org=(pos_x, pos_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), fontScale=1)
    else:
        cv2.putText(img=input_frame, text="Ghanaian", org=(pos_x, pos_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), fontScale=1)


while True:
    # grab the current frame
    ret, frame = vs.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    faces = faceCascade.detectMultiScale(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_found = frame[y:y + h, x:x + w]
        evaluate_img(model, face_found, frame, x, y)

    # show the frame to our screen
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # close all windows
cv2.destroyAllWindows()
