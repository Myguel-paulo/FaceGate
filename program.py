import cv2

imagem = cv2.imread("fotos/foto1.jpeg")

if imagem is None:
    print("Não foi possivel abrir a imagem")
    exit()

cv2.namedWindow("Foto", cv2.WINDOW_NORMAL) # Janela redimensionável
cv2.imshow("Foto", imagem)
cv2.waitKey(0) #Espera resposta do teclado
cv2.destroyAllWindows()

cv2.imwrite("fotos/copia.png", imagem)