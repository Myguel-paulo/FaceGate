import cv2
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

imagem = cv2.imread("fotos/foto1.jpeg")
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

face = face_classifier.detectMultiScale(
    cinza, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
)

for(x, y, w, h) in face:
    cv2.rectangle(imagem, (x,y), (x + w, y + h), (0, 255, 0), 4)

cv2.imwrite("fotos/imgComRetanguloAutomatico.jpeg", imagem)




cv2.namedWindow("Foto", cv2.WINDOW_NORMAL) # Janela redimensionável
cv2.imshow("Foto", imagem)
cv2.waitKey(0) #Espera resposta do teclado
cv2.destroyAllWindows()

# plt.figure(figsize=(20,10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()