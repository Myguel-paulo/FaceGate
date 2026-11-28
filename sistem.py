import cv2
import os
import pickle
import numpy as np
detector = cv2.FaceDetectorYN.create(
    "modelos/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320), 0.9
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

def carregar_cadastro(detector, reconhecedor, pasta="cadastro"):
    nomes = []
    embeddings = []

    for pessoa in os.listdir(pasta):
        pasta_pessoa = os.path.join(pasta, pessoa)
        if not os.path.isdir(pasta_pessoa):
            continue

        for arquivo in os.listdir(pasta_pessoa):
            imagem = cv2.imread(os.path.join(pasta_pessoa, arquivo))

            if imagem is None:
                continue

            embedding = embedding_do_rosto(imagem, detector, reconhecedor)

            if embedding is None:
                print("Foto ignorada:", pessoa, arquivo)
                continue

            nomes.append(pessoa)
            embeddings.append(embedding)
    return nomes, embeddings



def identificar(embedding, nomes, embeddings, reconhecedor, limiar= 0.363):
    if len(embeddings) == 0:
        return None

    notas = [reconhecedor.match(embedding, e, cv2.FaceRecognizerSF_FR_COSINE)
             for e in embeddings]

    melhor = int(np.argmax(notas))

    if notas[melhor] >= limiar:
        return nomes[melhor]
    return None

if os.path.exists("cadastro.pkl"):
    with open("cadastro.pkl", "rb") as f:
        nomes, embeddings = pickle.load(f)
else:
    nomes, embeddings = carregar_cadastro(detector, reconhecedor)
    with open("cadastro.pkl", "wb") as f:
        pickle.dump((nomes, embeddings), f)

camera = cv2.VideoCapture(1)

while True:
    ok, quadro = camera.read()
    if not ok:
        break

    quadro = cv2.flip(quadro, 1)
    altura, largura = quadro.shape[:2]
    detector.setInputSize((largura, altura))
    _,rostos = detector.detect(quadro)

    if rostos is not None:
        for rosto in rostos:
           alinhado = reconhecedor.alignCrop(quadro, rosto)
           embedding = reconhecedor.feature(alinhado)

           nome = identificar(embedding, nomes, embeddings, reconhecedor)
           acesso_liberado = nome is not None

           if acesso_liberado:
               texto = "Acesso liberado: " + nome
               cor = (0, 255, 0)
           else: 
               texto = "Acesso negado"
               cor = (0, 0, 255)

        x, y,w,h = rosto[:4].astype(int)
        cv2.rectangle(quadro, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(quadro, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        
    
    cv2.imshow("FaceGate", quadro)
    cv2.waitKey(1)
    if cv2.getWindowProperty("FaceGate", cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()


