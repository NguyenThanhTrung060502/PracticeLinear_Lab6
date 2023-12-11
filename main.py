import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загрузка изображения
img = Image.open('Luffy.png')

# Преобразование изображения к оттенкам серого
gray_img = img.convert('L')

# Представление изображения в виде матрицы
img_matrix = np.array(gray_img)

# SVD разложение матрицы
U, S, Vt = np.linalg.svd(img_matrix, full_matrices=False)

# Функция для восстановления изображения из SVD разложения сокращенного на k
def reconstruct_image(U, S, Vt, k):
    reconstructed_matrix = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return reconstructed_matrix

# Список значений k
k_values = [1, 20, 75, 100, 150, 200, 250, 300, 592]

# Изображения для различных значений k
plt.figure(figsize=(12, 8))
for i, k in enumerate(k_values, 1):
    new_image = reconstruct_image(U, S, Vt, k)
    plt.subplot(3, 3, i)
    plt.imshow(new_image, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.tight_layout()
plt.show()
