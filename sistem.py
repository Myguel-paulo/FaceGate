import cv2
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def detect_bounding_box(vid):
    imagemCinza = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

   
    face = face_classifier.detectMultiScale(
        imagemCinza, scaleFactor=1.1, minNeighbors=6, minSize=(110,110), maxSize=(450,450)
    )

    for(x, y, w, h) in face:
        cv2.rectangle(vid, (x,y), (x + w, y + h), (0, 255, 0), 4)


camera = cv2.VideoCapture(1)

while True:

    ok, quadro = camera.read()
    quadro = cv2.flip(quadro, 1)

    detect_bounding_box(quadro)
    cv2.imshow("Camera",quadro )
    cv2.waitKey(1)
    if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1 : break

camera.release()
cv2.destroyAllWindows()



