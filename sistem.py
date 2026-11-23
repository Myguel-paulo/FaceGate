import cv2

detector = cv2.FaceDetectorYN.create(
    "modelos/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320), 0.8
)

imagem = cv2.imread("fotos/foto1.jpeg")

if imagem is None:
    print("Não foi possivel abrir a imagem")
    exit()


altura, largura = imagem.shape[:2]
detector.setInputSize((largura,altura))

_,rostos = detector.detect(imagem)


if rostos is None:
    print("Nenhum rosto encontrado")
else:
    print("Rostos encontrados:", len(rostos))

    for rosto in rostos:

        x,y,w,h = rosto[:4].astype(int)
        cv2.rectangle(imagem, (x,y), (x + w, h + y), (0, 255, 0), 4)

def mostrar(nome, img, altura_max=800):
        altura, largura = img.shape[:2]
        escala = min(altura_max / altura, 1)
        menor = cv2.resize(img, (int(largura * escala), int(altura * escala)))
        cv2.imshow(nome, menor)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

mostrar("Rostos", imagem)

