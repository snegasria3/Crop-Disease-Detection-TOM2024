import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL_PATH = "efficientnet_best.pth"
SUGGESTION_PATH = "disease_suggestions.json"

CLASSES = sorted([
    'maize_diseases_Abiotic_disease-D','maize_diseases_Curvularia-D','maize_diseases_Healthy_leaf',
    'maize_diseases_Helminthosporiosis-D','maize_diseases_Rust-D','maize_diseases_Stripe-D','maize_diseases_Virosis-D',
    'maize_pests_Chenille l├⌐gionnaire-P','maize_pests_Pucerons-P','maize_pests_activities_Chenille l├⌐gionnaire- A',
    'onion_diseases_Alternaria_D','onion_diseases_Bulb_blight-D','onion_diseases_Fusarium-D','onion_diseases_Healthy_leaf',
    'onion_diseases_Virosis-D','onion_pests_Chenilles-P','tomato_diseases_Bacterial_floundering_d',
    'tomato_diseases_Blossom_end_rot_d','tomato_diseases_Mite_d','tomato_diseases_alternaria_d',
    'tomato_diseases_alternaria_mite_d','tomato_diseases_exces_nitrogen_d','tomato_diseases_fusarium_d',
    'tomato_diseases_healthy_fruit','tomato_diseases_healthy_leaf','tomato_diseases_sunburn_d',
    'tomato_diseases_tomato_late_blight_d','tomato_diseases_virosis_d','tomato_pests_helicoverpa_armigera_p',
    'tomato_pests_tuta_absoluta_p'
])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")

# ---------------- MODEL ----------------
model = efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(1280, 512),
    nn.SiLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.3),
    nn.Linear(512, len(CLASSES))
)
model.to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- SUGGESTIONS ----------------
with open("disease_suggestions.json", encoding="utf-8") as f:
    disease_suggestions = json.load(f)


# ---------------- HELPERS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_class = CLASSES[pred_idx]
            # Fetch suggestion from JSON
            suggestion = disease_suggestions.get(pred_class, {}).get("suggestion", "No suggestion available.")

        return render_template("result.html",
                               filename=filename,
                               prediction=pred_class,
                               suggestion=suggestion)
    return redirect(url_for("index"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)

if __name__ == "__main__":
    app.run(debug=True)
