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

#Deixando a imagem cinza
imgCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imwrite("fotos/imgCinza.jpeg", imgCinza)

#Suavizando a imagem cinza
imgSuavizada = cv2.GaussianBlur(imgCinza, (11,11),0)
cv2.imwrite("fotos/imgSuavizada.jpeg", imgSuavizada)


imgComRetangulo = imagem.copy()

#Aplicando retângulo na imagem
cv2.rectangle(imgComRetangulo, (50, 50), (200, 200), (0, 255,0), 2)
cv2.imwrite("fotos/imgComRetangulo.jpeg", imgComRetangulo)

#Aplicando texto na imagem

cv2.putText(imgComRetangulo,"Hello world", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
            (0, 255, 0), 2)

cv2.imwrite("fotos/imgRetanguloComTexto.jpeg", imgComRetangulo)