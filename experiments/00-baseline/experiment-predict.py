import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "experiments/00-baseline/model.keras"
CLASSES_PATH = "experiments/00-baseline/classes.json"
IMAGE_PATH = "experiments/00-baseline/my_plant.jpg"
IMAGE_SIZE = (150, 150)

def main():
    # 1. Завантажуємо модель та класи
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 2. Завантажуємо та обробляємо фото
    img = tf.keras.utils.load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Виконуємо прогноз
    predictions = model.predict(img_batch)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0])) * 100

    # 4. Виводимо результат
    print(f"Зображення: {IMAGE_PATH}")
    print(f"Передбачений клас: {predicted_class}")
    print(f"Впевненість моделі: {confidence:.2f}%")

if __name__ == "__main__":
    main()