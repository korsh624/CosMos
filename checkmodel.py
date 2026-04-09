import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# =========================
# 1. ЗАГРУЗКА МОДЕЛИ
# =========================
model = tf.keras.models.load_model("eurosat_model.h5")

# =========================
# 2. СПИСОК КЛАССОВ
# =========================
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# (опционально — перевод для детей)
class_names_ru = {
    'AnnualCrop': 'Поля 🌾',
    'Forest': 'Лес 🌲',
    'HerbaceousVegetation': 'Растительность 🌿',
    'Highway': 'Дорога 🛣️',
    'Industrial': 'Заводы 🏭',
    'Pasture': 'Пастбище 🐄',
    'PermanentCrop': 'Сады 🌳',
    'Residential': 'Город 🏙️',
    'River': 'Река 🌊',
    'SeaLake': 'Море / Озеро 🌊'
}

# =========================
# 3. ЗАГРУЗКА КАРТИНКИ
# =========================
img_path = "test.jpg"  # 👈 поменяй на свою картинку

img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)

# добавляем размер батча
img_array = np.expand_dims(img_array, axis=0)

# =========================
# 4. ПРЕДСКАЗАНИЕ
# =========================
pred = model.predict(img_array)

# переводим в вероятности
score = tf.nn.softmax(pred[0])

# =========================
# 5. ВЫВОД РЕЗУЛЬТАТА
# =========================
predicted_class = class_names[np.argmax(score)]
confidence = np.max(score)

print("📸 Картинка:", img_path)
print("🧠 Модель думает:")

# если есть перевод
if predicted_class in class_names_ru:
    print("👉", class_names_ru[predicted_class])
else:
    print("👉", predicted_class)

print(f"📊 Уверенность: {confidence*100:.2f}%")