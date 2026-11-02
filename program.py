import cv2

#Leitura da imagem
imagem = cv2.imread("fotos/foto1.jpeg")

if imagem is None:
    print("Não foi possivel abrir a imagem")
    exit()

#Abrindo a imagem em uma janela
cv2.namedWindow("Foto", cv2.WINDOW_NORMAL) # Janela redimensionável
cv2.imshow("Foto", imagem)
cv2.waitKey(0) #Espera resposta do teclado
cv2.destroyAllWindows()

#Salvando copia da imagem
cv2.imwrite("fotos/copia.jpeg", imagem)

#Deixando a imagem cinzaa
imgCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imwrite("fotos/imgCinza.jpeg", imgCinza)

#Suavizando a imagem cinza
imgSuavizada = cv2.GaussianBlur(imgCinza, (11,11),0)
cv2.imwrite("fotos/imgSuavizada.jpeg", imgSuavizada)


imgComRetangulo = imagem.copy()

cv2.rectangle(imgComRetangulo, (100, 110), (200, 200), (0, 255,0), 0)

cv2.namedWindow("Imagem com Retângulo", cv2.WINDOW_NORMAL)
cv2.imshow("Imagem com Retângulo", imgComRetangulo)
cv2.waitKey(0)
cv2.destroyAllWindows()