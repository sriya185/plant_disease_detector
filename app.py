from flask import Flask, render_template, request, url_for, send_file
import cv2
import numpy as np
import os
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Helper: Remove unsupported characters (emojis, symbols)
def remove_emojis(text):
    """Removes non-latin1 characters that FPDF cannot encode."""
    return text.encode('latin-1', 'ignore').decode('latin-1')


# 🌿 Enhanced Disease Detection Function
def predict_disease(image_path):
    """Enhanced plant disease detection using color and texture analysis."""
    img = cv2.imread(image_path)
    if img is None:
        return "Error", "No Image Detected", None, "No treatment available"

    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask for green (healthy)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (256 * 256)

    # Mask for yellow (nutrient deficiency)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (256 * 256)

    # Mask for brown (fungal)
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([25, 255, 180])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (256 * 256)

    # Combine masks
    infection_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    kernel = np.ones((3, 3), np.uint8)
    infection_mask = cv2.morphologyEx(infection_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Highlight infected regions in red
    output = img.copy()
    output[infection_mask > 0] = [0, 0, 255]

    processed_path = os.path.join(UPLOAD_FOLDER, "processed_leaf.jpg")
    cv2.imwrite(processed_path, output)

    # Adjust thresholds by brightness
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    adjusted_threshold = 0.04 if brightness > 100 else 0.03

    # Decision logic
    if brown_ratio > adjusted_threshold:
        disease_name = "Fungal Infection (Black Spot / Rust Disease)"
        status = "Infected Leaf"
        treatment = (
            "Use organic fungicides like neem oil or copper-based sprays. "
            "Remove infected leaves and ensure proper drainage to prevent spread."
        )
    elif yellow_ratio > adjusted_threshold:
        disease_name = "Nutrient Deficiency (Chlorosis)"
        status = "Infected Leaf"
        treatment = (
            "Add nitrogen or iron-rich fertilizer and ensure regular watering. "
            "Enhance soil quality with organic compost."
        )
    elif green_ratio > 0.85:
        disease_name = "Healthy Leaf"
        status = "Healthy Leaf"
        treatment = (
            "The plant is healthy! Maintain balanced watering, sunlight, and nutrient supply."
        )
    else:
        disease_name = "Mild Stress or Uncertain Condition"
        status = "Monitor Leaf"
        treatment = (
            "Could be early stress. Avoid overwatering and check for pests."
        )

    return status, disease_name, processed_path, treatment


# 🌾 Main Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", prediction="No file uploaded.")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction, disease_type, processed_path, treatment = predict_disease(filepath)

    original_image = url_for("static", filename=f"uploads/{file.filename}")
    processed_image = url_for("static", filename="uploads/processed_leaf.jpg")

    return render_template(
        "index.html",
        prediction=prediction,
        disease_type=disease_type,
        treatment=treatment,
        original_image=original_image,
        processed_image=processed_image,
        file_name=file.filename,
    )


# 📄 PDF Download Route
@app.route("/download_report/<filename>")
def download_report(filename):
    """Generate and download a clean, readable PDF report."""
    original_image_path = os.path.join(UPLOAD_FOLDER, filename)
    processed_image_path = os.path.join(UPLOAD_FOLDER, "processed_leaf.jpg")

    # Recalculate details for consistency
    prediction, disease_type, _, treatment = predict_disease(original_image_path)

    pdf_path = os.path.join(UPLOAD_FOLDER, "Plant_Report.pdf")

    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, remove_emojis("Plant Disease Detection Report"), ln=True, align="C")
    pdf.ln(8)

    # Basic Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Disease: {remove_emojis(disease_type)}", ln=True)
    pdf.cell(0, 10, f"Diagnosis: {remove_emojis(prediction)}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Treatment Advice: {remove_emojis(treatment)}", align="L")
    pdf.ln(8)

    # Add Images
    if os.path.exists(original_image_path):
        pdf.image(original_image_path, x=15, y=None, w=80)
    if os.path.exists(processed_image_path):
        pdf.image(processed_image_path, x=110, y=pdf.get_y() - 60, w=80)

    pdf.ln(85)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, remove_emojis("Generated by Smart Plant Disease Detection System"), ln=True, align="C")

    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
