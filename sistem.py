import cv2

detector = cv2.FaceDetectorYN.create(
    "modelos/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320), 0.7
)
reconhecedor = cv2.FaceRecognizerSF.create(
    "modelos/face_recognition_sface_2021dec.onnx", ""
)
def embedding_do_rosto(imagem, detector, reconhecedor):
    altura, largura = imagem.shape[:2]
    detector.setInputSize((largura, altura))
    _, rostos = detector.detect(imagem)

    if rostos is None or len(rostos) != 1:
        return None

    alinhado = reconhecedor.alignCrop(imagem, rostos[0])
    return reconhecedor.feature(alinhado)

foto_a = cv2.imread("fotos/reconhecimentoDeRostosComDNN/foto1.jpeg")
foto_b = cv2.imread("fotos/reconhecimentoDeRostosComDNN/foto2.jpeg")

emb_a = embedding_do_rosto(foto_a, detector, reconhecedor)
emb_b = embedding_do_rosto(foto_b, detector, reconhecedor)

nota = reconhecedor.match(emb_a, emb_b, cv2.FaceRecognizerSF_FR_COSINE )

print(nota)
if nota < 0.369:
    print("Acesso Negado")
else:
    print("Acesso Liberado")
# camera = cv2.VideoCapture(1)

# while True:
#     ok, quadro = camera.read()
#     if not ok:
#         break

#     altura, largura = quadro.shape[:2]
#     detector.setInputSize((largura, altura))
#     _,rostos = detector.detect(quadro)

#     if rostos is not None:
#         for rosto in rostos:
#             x,y,w,h = rosto[:4].astype(int)
#             cv2.rectangle(quadro, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     quadro = cv2.flip(quadro, 1)
#     cv2.imshow("FaceGate", quadro)
#     cv2.waitKey(1)
#     if cv2.getWindowProperty("FaceGate", cv2.WND_PROP_VISIBLE) < 1:
#         break

# camera.release()
# cv2.destroyAllWindows()


