import cv2, os

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'Seesh'

path = os.path.join(datasets, sub_data)
os.makedirs(path, exist_ok=True)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)  # Change to 1 or 2 if needed
if not webcam.isOpened():
    print("Error: Camera not found!")
    exit()

count = 1
while count <= 30:
    ret, im = webcam.read()
    if not ret or im is None:
        print("Failed to capture image. Retrying...")
        continue

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_resize = cv2.resize(gray[y:y + h, x:x + w], (width, height))
        cv2.imwrite(f"{path}/{count}.png", face_resize)
        count += 1

    cv2.imshow('Face Capture', im)
    if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
