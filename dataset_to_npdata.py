import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense





"""
1. kısım veri setini istenilen boyuta dönüştürme 28x28 grayscale gibi
"""

# Veri seti klasör yolunu belirtin
dataset_path = './dataset'

# Veri ve etiketleri saklayacak listeler
data = []
labels = []

# Her klasörün içinde gezin (1, 2, 3, 4, 5 gibi)
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):  # Sadece klasörleri işle
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            # Görüntüyü yükleyin
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlama
            # Görüntüyü boyutlandırın
            resized_image = cv2.resize(image, (28, 28))  # 28x28 piksele küçültme
            # Görüntüleri ve etiketlerini listeye ekleyin
            data.append(resized_image)
            labels.append(int(label))  # Klasör adı etiket olarak kullanılıyor

# Veri ve etiketleri numpy dizilerine çevirin
data = np.array(data, dtype='float32') / 255.0  # Normalizasyon (0-1 aralığına)
data = np.expand_dims(data, axis=-1)  # Kanal boyutunu ekleyin (28x28x1)
labels = np.array(labels)

print(f"Toplam görüntü: {len(data)}, Toplam etiket: {len(labels)}")

"""
2. kısım %80 eğitim ve %20 test verisine ayırma
"""

# Veriyi %80 eğitim, %20 test olacak şekilde ayırma
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Eğitim seti boyutu: {x_train.shape}, Test seti boyutu: {x_test.shape}")


"""
3. kısım model oluşturma
"""

# Model tanımlama
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # İlk Convolutional katman
    MaxPooling2D((2, 2)),  # MaxPooling katmanı
    Conv2D(64, (3, 3), activation='relu'),  # İkinci Convolutional katman
    MaxPooling2D((2, 2)),
    Flatten(),  # Tam bağlantılı katmanlar için düzleştirme
    Dense(128, activation='relu'),  # Gizli katman
    Dense(6, activation='softmax')  # Çıkış katmanı (6 sınıf: 0, 1, 2, 3, 4, 5)
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model özetini görüntüleme
model.summary()

"""
4. kısım modeli eğitme
"""

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

"""
5.kısım modeli test etme 
"""

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu: {test_accuracy * 100:.2f}%")

"""
6. kısım modeli kullanarak tahminde bulunma
"""

# Yeni bir görüntü yükleme
new_image = cv2.imread('./4parmak.jpg', cv2.IMREAD_GRAYSCALE)
# new_image = cv2.imread('./dataset/3/WIN_20241213_10_38_30_Pro.jpg', cv2.IMREAD_GRAYSCALE)
new_image_resized = cv2.resize(new_image, (28, 28)) / 255.0
new_image_resized = np.expand_dims(new_image_resized, axis=(0, -1))  # Model için boyut ekleme

# Tahmin
prediction = model.predict(new_image_resized)
predicted_label = np.argmax(prediction)
print(f"Model tahmini: {predicted_label}")
