import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging

# ====================================
# CONFIG
# ====================================

MODEL_PATH = "model/distilbert_final_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================
# LOAD MODEL ON STARTUP
# ====================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)

print("Number of labels:", model.config.num_labels)
print("Model name:", model.config._name_or_path)


# Print label mapping
print("Model label mapping:", model.config.id2label)

# Detect which index corresponds to Toxic
id2label = model.config.id2label

# Default assumption
TOXIC_INDEX = 1  

# Try automatic detection
for idx, label in id2label.items():
    if "toxic" in label.lower():
        TOXIC_INDEX = idx

print("Detected Toxic class index:", TOXIC_INDEX)

# ====================================
# FLASK APP
# ====================================

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Toxic Language Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in request body"}), 400

    text = data["text"]

    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)

    # Debug (optional: remove after testing)
    print("Logits:", logits)
    print("Probabilities:", probabilities)

    toxic_probability = probabilities[0][TOXIC_INDEX].item()

    prediction = "Toxic" if toxic_probability >= 0.5 else "Non-Toxic"

    return jsonify({
        "toxicity_probability": round(toxic_probability, 4),
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
