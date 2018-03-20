# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
import os
import numpy as np
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.detectors import FaceDetector
import face_recognition_system.operations as op
import cv2
from cv2 import __version__
import pyttsx
from translation import baidu, google, youdao, iciba


def get_images(frame, faces_coord, shape):
    """ Perfrom transformation on original and face images.

    This function draws the countour around the found face given by faces_coord
    and also cuts the face from the original image. Returns both images.

    :param frame: original image
    :param faces_coord: coordenates of a rectangle around a found face
    :param shape: indication of which shape should be drwan around the face
    :type frame: numpy array
    :type faces_coord: list of touples containing each face information
    :type shape: String
    :return: two images containing the original plus the drawn contour and
             anoter one with only the face.
    :rtype: a tuple of numpy arrays.
    """
    if shape == "rectangle":
        faces_img = op.cut_face_rectangle(frame, faces_coord)
        frame = op.draw_face_rectangle(frame, faces_coord)
    elif shape == "ellipse":
        faces_img = op.cut_face_ellipse(frame, faces_coord)
        frame = op.draw_face_ellipse(frame, faces_coord)
    faces_img = op.normalize_intensity(faces_img)
    faces_img = op.resize(faces_img)
    return (frame, faces_img)

def say(s):     
	engine = pyttsx.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', 150)
        voices= engine.getProperty('voices')
        #for voice in voices:                                                                                    
        engine.setProperty('voice', 'english-us')
        #print voice.id                                                                                          
        engine.say(s)
        a = engine.runAndWait() #blocks     


people_folder = "face_recognition_system/people/"
shape = "ellipse"

try:
        people = [person for person in os.listdir(people_folder)]
except:
        print "Have you added at least one person to the system?"
        sys.exit()
print "This are the people in the Recognition System:"
for person in people:
        print "-" + person

    
detector = FaceDetector('face_recognition_system/frontal_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
threshold = 105
images = []
labels = []
labels_people = {}
for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
try:
        recognizer.train(images, np.array(labels))
except:
        print "\nOpenCV Error: Do you have at least two people in the database?\n"
        sys.exit()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > .75 :
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = CLASSES[idx]
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			if label == "person":
				detector = FaceDetector('face_recognition_system/frontal_face.xml')
				threshold = 105
				people_folder = "face_recognition_system/people/"
				shape = "ellipse"
				faces_coord = detector.detect(frame, False)
				if len(faces_coord):
	            			frame, faces_img = get_images(frame, faces_coord, shape)
					for i, face_img in enumerate(faces_img):
	                			if __version__ == "3.1.0":
	                    				collector = cv2.face.MinDistancePredictCollector()
	                    				recognizer.predict(face_img, collector)
	                    				conf = collector.getDist()
	                    				pred = collector.getLabel()
	                			else:
	                    				pred, conf = recognizer.predict(face_img)
	                			print "Prediction: " + str(pred)
	                			print 'Confidence: ' + str(round(conf))
	                			print 'Threshold: ' + str(threshold)
						if conf < threshold:
		 					say(labels_people[pred].capitalize())
		    					cv2.putText(frame, labels_people[pred].capitalize(), (startX, y),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			else: 
				say(label.capitalize())
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
