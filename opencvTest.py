#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
emotion_class = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
best_model = load_model('expression.model')

def detect_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_AREA)
    face = face.reshape(1, 48, 48, 1)
    face = face.astype(np.float32)
    face = face / 255
    emotion_index = best_model.predict_classes(face)

    return emotion_class[emotion_index[0]]


# In[ ]:


FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
CAP = cv2.VideoCapture(0)
w = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))  
h = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('out.avi',fourcc, 30.0, (w,h))

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (100, 200, 100)
LINE_TYPE = 2

while 1:
    RET, I = CAP.read()

    G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    FACES = FACE_DETECTOR.detectMultiScale(G, 1.1, 3)

    for (x, y, w, h) in FACES:
        cv2.rectangle(I, (x - 10, y - 30), (x + w + 10, y + h + 30), (255, 100, 100), 3)
        Face = I[y:y+h, x:x+h]
        title = detect_emotion(Face)
        cv2.putText(I, title,
                    (x+10, y-5),
                    FONT,
                    FONT_SCALE,
                    FONT_COLOR,
                    LINE_TYPE)

    cv2.imshow('I', I)
    KEY = cv2.waitKey(60)
    out.write(I)
    if KEY == ord('q'):
        break


CAP.release()
out.release()
cv2.destroyAllWindows()

