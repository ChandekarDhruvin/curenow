import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import pickle
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = FastAPI(title="Medical Disease Prediction API", version="1.0")

# -------------------------------- IMAGE-BASED DISEASE PREDICTION -------------------------------- #

class MultiHeadMedicalModel(nn.Module):
    def __init__(self, num_brain_classes=4, num_lung_classes=4, num_skin_classes=9):
        super().__init__()
        # Backbone for grayscale images (brain & lung)
        self.backbone_gray = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone_gray.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Backbone for RGB images (skin)
        self.backbone_rgb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze early layers, unfreeze last 3 layers
        for param in self.backbone_gray.parameters():
            param.requires_grad = False
        for param in self.backbone_rgb.parameters():
            param.requires_grad = False
        for param in self.backbone_gray.features[-3:].parameters():
            param.requires_grad = True
        for param in self.backbone_rgb.features[-3:].parameters():
            param.requires_grad = True

        # Classification heads
        self.brain_classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1280, num_brain_classes)  # Use index 1 for weight keys to match the saved state_dict
        )

        self.lung_classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1280, num_lung_classes)
        )

        self.skin_classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1280, num_skin_classes)
        )

    def forward(self, x, task):
        if task in ["brain", "lung"]:
            x = self.backbone_gray.features(x)
            x = self.backbone_gray.avgpool(x)
        else:
            x = self.backbone_rgb.features(x)
            x = self.backbone_rgb.avgpool(x)

        x = torch.flatten(x, 1)
        return {
            "brain": self.brain_classifier(x),
            "lung": self.lung_classifier(x),
            "skin": self.skin_classifier(x)
        }[task]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadMedicalModel(num_brain_classes=4, num_lung_classes=4, num_skin_classes=9).to(device)
model.load_state_dict(torch.load("MRI and XRAY classifier/Model/Multi_head_medical_model.pth", map_location=device))
model.eval()

# Labels
task_labels = {
    "brain": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "lung": ["Corona Virus Disease", "Normal", "Pneumonia", "Tuberculosis"],
    "skin": ["Actinic keratosis", "Dermatofibroma", "Squamous cell carcinoma", "Atopic Dermatitis",
             "Melanocytic nevus", "Tinea Ringworm Candidiasis", "Benign keratosis", "Melanoma", "Vascular lesion"]
}

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/image_disease_classification/")
async def predict_image(task: str, file: UploadFile = File(...)):
    if task not in task_labels:
        return {"error": f"Invalid task. Choose from: {list(task_labels.keys())}"}

    image = Image.open(io.BytesIO(await file.read()))
    if task in ["brain", "lung"]:
        image = image.convert("L")  # Convert to grayscale

    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image, task)
        probs = F.softmax(output, dim=1)
        predicted_label = task_labels[task][torch.argmax(probs).item()]

    return {"task": task, "predicted_class": predicted_label}

# -------------------------------- TEXT-BASED DISEASE PREDICTION -------------------------------- #

# Hugging Face model repo details
MODEL_PATH = "Symptoms Analyzer/Model/fine_tuned_pubmedbert"

# Load model and tokenizer
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
text_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
classifier = pipeline("text-classification", model=text_model, tokenizer=text_tokenizer)

# Load label encoder
with open("Symptoms Analyzer/Model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/symptom_disease_prediction/")
def predict_text(input_data: SymptomInput):
    result = classifier(input_data.symptoms)
    predicted_label = label_encoder.inverse_transform([int(result[0]["label"].replace("LABEL_", ""))])[0]
    return {"predicted_disease": predicted_label}
