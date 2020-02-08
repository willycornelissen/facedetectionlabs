import cv2
import face_recognition
from imutils.video import VideoStream
from imutils import face_utils
import dlib
from PIL import Image, ImageDraw
import numpy as np

def detect_faces(image):
  boxes=face_recognition.face_locations(image)
  for (top, right, bottom, left) in boxes:
    cv2.rectangle(image, (left,top), (right,bottom), (0,255,0),2)

def detect_landmarks(image):
  gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)
  for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    for (x, y) in shape:
      cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def add_mustache(image):
  imgMustache = cv2.imread("images/mustache.png", -1)
  orig_mask = imgMustache[:,:,3]
  orig_mask_inv = cv2.bitwise_not(orig_mask)
  imgMustache = imgMustache[:,:,0:3]
  origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
  # Find all facial features in all the faces in the image
  face_landmarks_list = face_recognition.face_landmarks(image)
  for face_landmarks in face_landmarks_list:
    mustacheWidth = abs(3 * (face_landmarks['nose_tip'][0][0] - face_landmarks['nose_tip'][-1][0]))
    mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth)
    mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
    y1=int(face_landmarks['nose_tip'][0][1] - (mustacheHeight/2)) + 10
    y2=int(y1+mustacheHeight)
    x1=int(face_landmarks['nose_tip'][-1][0] - (mustacheHeight/2)) - 30
    x2=int(x1+mustacheWidth)
    roi = image[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
    image[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)


faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

#Dlib actions
video=VideoStream(src=0).start()
predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
font = cv2.FONT_HERSHEY_SIMPLEX
pace=1

while True:
    frame=video.read()
    if (pace==1):
      # Simple Face Detection
      detect_faces(frame)
      cv2.putText(frame,'Face Detection',(20,30), font, 1,(255,255,255),2)
    if (pace==2):
      # Facial Landmarks detection
      detect_landmarks(frame)
      cv2.putText(frame,'Face Landmarks',(20,30), font, 1,(255,255,255),2)
    if (pace==3):
      add_mustache(frame)
      cv2.putText(frame,'Mustache',(20,30), font, 1,(255,255,255),2)
    cv2.imshow("Face", frame)
    key=cv2.waitKey(3)
    if key == ord("q"):
      break
    if key == ord("n"):
      pace=pace + 1









