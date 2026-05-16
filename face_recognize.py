import cv2, numpy, os

size = 4
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
for subdir in os.listdir(datasets):
    subjectpath = os.path.join(datasets, subdir)
    if os.path.isdir(subjectpath):
        names[id] = subdir
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            images.append(cv2.imread(path, 0))
            labels.append(id)
        id += 1

if len(images) == 0:
    print("Error: No training images found!")
    exit()

(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Use LBPH if FisherFace doesn't work
try:
    model = cv2.face.FisherFaceRecognizer_create()
except AttributeError:
    model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error: Haar cascade file not found!")
    exit()

webcam = cv2.VideoCapture(0)  # Change if needed

if not webcam.isOpened():
    print("Error: Camera not found!")
    exit()

cnt = 0
while True:
    ret, im = webcam.read()
    if not ret or im is None:
        print("Error: Failed to capture image!")
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        confidence = prediction[1]

        if confidence < 800:
            cv2.putText(im, f'{names[prediction[0]]} - {confidence:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person Detected!")
                cv2.imwrite("unknown.jpg", im)
                cnt = 0

    cv2.imshow('Face Recognition', im)
    if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
