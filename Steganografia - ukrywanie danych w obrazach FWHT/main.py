import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from PIL import Image

# wykorzystywane obrazy konwertuję z RGB do skali szarości

lena_tmp = Image.open('lena.png').convert('L')
agh_tmp = Image.open('agh.png').convert('L')

# otworzone obrazy zapisuję w macierzach

lena = np.array(lena_tmp)
agh = np.array(agh_tmp)

# zmieniam typ macierzy na float

nosnik = lena.astype(float)
dolaczany = agh.astype(float)

# na każdym etapie wyświetlam pierwszy element macierzy żeby zobaczyć co dzieje się z danymi

print("lena:")
print(lena[0, 0])
print(end='\n')
print("agh:")
print(agh[0, 0])
print(end='\n')

# definicja funkcji transformującej


def wht(x):
    h = hadamard(x.shape[0], dtype=float)
    y = h @ x @ h
    y = y / x.shape[0]
    return y

# transformacja odwrotna = trnsformacja ale czytelności kodu zdefiniowałem funkcję o innej nazwie


def iwht(x):
    h = hadamard(x.shape[0], dtype=float)
    y = h @ x @ h
    y = y / x.shape[0]
    return y


transform_nosnik = wht(lena)
print("lena po transforamcji:")
print(transform_nosnik[0, 0])
print(end='\n')

transform_dolaczany = wht(agh)
print("agh po transforamcji:")
print(transform_dolaczany[0, 0])
print(end='\n')

transform_stegokontener = transform_nosnik + 0.01*transform_dolaczany
print("lena po transforamcji i modyfikacji współczynników trasnformacją agh*0.001:")
print(transform_stegokontener[0, 0])
print(end='\n')

stegokontener = iwht(transform_stegokontener)
print("lena wraz z ukrytą informacją po transformacji odwrotnej:")
print(stegokontener[1, 1])
print(end='\n')

print("odzyskanie ukrytej informacji z lena:")

transform_stegokontener_2 = wht(stegokontener)
print("lena z ukrytą informacją po transformacji:")
print(transform_stegokontener_2[0, 0])
print(end='\n')

transform_dolaczany_2 = (transform_stegokontener_2 - transform_nosnik)*100
print("(lena z ukrytą informacją po transformacji minus lena po transformacji)*100:")
print(transform_dolaczany_2[0, 0])
print(end='\n')

dolaczany_2 = iwht(transform_dolaczany_2)
print("transformacja odwrotna wyniku odejmowania:")
print(dolaczany_2[0, 0])
print(end='\n')

# wyświetlam wynik działania programu

plt.subplot(221)
plt.imshow(nosnik, cmap='gray')
plt.subplot(222)
plt.imshow(dolaczany, cmap='gray')
plt.subplot(223)
plt.imshow(stegokontener, cmap='gray')
plt.subplot(224)
plt.imshow(dolaczany_2, cmap='gray')
plt.show()
