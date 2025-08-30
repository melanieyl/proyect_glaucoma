from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import os, json

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Carga de modelo (sin compilar para inferencia)
MODEL_PATH = "ml/sequential_glaucoma.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

_, H, W, C = model.input_shape
IS_RGB = (C == 3)

LABELS = None
labels_path = "ml/labels.json"
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        LABELS = json.load(f)  # p.ej: ["Normal", "Glaucoma Temprano", "Glaucoma Avanzado"]

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(filepath: str) -> np.ndarray:
    # cv2.imread lee BGR
    img = cv2.imread(filepath, cv2.IMREAD_COLOR if IS_RGB else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la imagen. Verifica el archivo.")
    # Resize: cv2 usa (W,H)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    if IS_RGB:
        # Convertir BGR->RGB para coherencia con tf/keras
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Expandir canal si el modelo es 1 canal
        img = np.expand_dims(img, axis=-1)

    img = img.astype("float32") / 255.0
    # Añadir batch
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No se envió archivo", 400

    file = request.files["file"]
    if file.filename == "":
        return "Archivo vacío", 400
    if not allowed_file(file.filename):
        return "Extensión no permitida", 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        x = preprocess_image(path)
        probs = model.predict(x, verbose=0)[0]   # vector de probabilidades
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if LABELS and 0 <= idx < len(LABELS):
            label = LABELS[idx]
        else:
            label = f"Clase {idx}"

        # Respuesta HTML (compatible con tu front actual)
        return f"<h3>Predicción: {label}</h3><p>Confianza: {conf:.2%}</p>"

        # Si prefieres JSON:
        # return jsonify({"label": label, "confidence": conf, "probs": probs.tolist()})

    except Exception as e:
        return f"Error en predicción: {str(e)}", 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
