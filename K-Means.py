import numpy as np
import matplotlib.pyplot as plt
import cv2

# Membaca gambar, sesuaikan path dengan lokasi gambar Anda
image = cv2.imread('F:/My Drive/KULIAH/Semester 4/06. Pengolahan Citra/UAS/images/image.jpeg') 

# Mengubah warna gambar dari BGR ke RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Menampilkan gambar asli
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')
plt.show()

# Membentuk ulang gambar menjadi susunan piksel 2D dengan 3 nilai warna (RGB)
pixel_vals = image.reshape((-1,3))

# Mengonversi tipe data menjadi float
pixel_vals = np.float32(pixel_vals)

# Menentukan kriteria untuk algoritme k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Menentukan jumlah cluster
k = 3

# Melakukan k-means clustering
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Mengonversi data menjadi nilai 8-bit
centers = np.uint8(centers)

# Membentuk ulang data menjadi dimensi gambar asli
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))

# Menampilkan gambar hasil segmentasi
plt.imshow(segmented_image)
plt.title('Gambar Tersegmentasi')
plt.axis('off')
plt.show()
