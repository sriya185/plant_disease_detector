import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Simulated class labels (same as your dataset)
classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Simulate prediction
def predict_disease(image_path):
    predicted_label = random.choice(classes)
    confidence = round(random.uniform(85, 99), 2)

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Detected: {predicted_label} ({confidence}%)", color='red', fontsize=12)
    plt.show()

    print(f"✅ Prediction: {predicted_label} | Confidence: {confidence}%")

if __name__ == "__main__":
    img_path = "data/leaf4.jpg"  # change to your test image
    predict_disease(img_path)
