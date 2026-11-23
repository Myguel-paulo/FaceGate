import cv2

detector = cv2.FaceDetectorYN.create(
    "modelos/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320), 0.8
)

camera = cv2.VideoCapture(1)

while True:
    ok, quadro = camera.read()
    if not ok:
        break

    altura, largura = quadro.shape[:2]
    detector.setInputSize((largura, altura))
    _,rostos = detector.detect(quadro)

    if rostos is not None:
        for rosto in rostos:
            x,y,w,h = rosto[:4].astype(int)
            cv2.rectangle(quadro, (x, y), (x + w, y + h), (0, 255, 0), 2)

    quadro = cv2.flip(quadro, 1)
    cv2.imshow("FaceGate", quadro)
    cv2.waitKey(1)
    if cv2.getWindowProperty("FaceGate", cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()


