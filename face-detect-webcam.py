import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier('./assets/haarcascade_frontalface_alt.xml')

vs = cv2.VideoCapture(0)

vs.set(3, 1280)
vs.set(4, 1024)

ethnicity_model = load_model('./assets/ethnicity_model')

# emotion_model = load_model('./assets/emotion_model')

IMG_SIZE = 300


def evaluate_img_ethnicity(input_model, img, input_frame, pos_x, pos_y):
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


def evaluate_img_emotion(input_model, img, input_frame, pos_x, pos_y):
    img = Image.fromarray(img, 'RGB')
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = np.array(img)
    result = np.array([img.reshape(IMG_SIZE, IMG_SIZE, 1)])
    result = input_model.predict(result, verbose=0)
    if result[0][0] > 0.5:
        cv2.putText(img=input_frame, text="Happy", org=(pos_x, pos_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), fontScale=1)
    else:
        cv2.putText(img=input_frame, text="Sad", org=(pos_x, pos_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 255), fontScale=1)


while True:
    ret, frame = vs.read()

    if frame is None:
        break

    faces = faceCascade.detectMultiScale(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_found = frame[y:y + h, x:x + w]

        evaluate_img_ethnicity(ethnicity_model, face_found, frame, x, y)
        # evaluate_img_emotion(emotion_model, face_found, frame, x, y)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
