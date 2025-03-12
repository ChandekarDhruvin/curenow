import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# -------------------------------- DEFINE THE MODEL -------------------------------- #
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
            nn.Linear(1280, num_brain_classes)  
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

# -------------------------------- LOAD MODEL & WEIGHTS -------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadMedicalModel(num_brain_classes=4, num_lung_classes=4, num_skin_classes=9).to(device)

# Load weights
model_path = "x-rayy\Multi_head_medical_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully and is in evaluation mode.")

# -------------------------------- TASK LABELS -------------------------------- #
task_labels = {
    "brain": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "lung": ["Corona Virus Disease", "Normal", "Pneumonia", "Tuberculosis"],
    "skin": ["Actinic keratosis", "Dermatofibroma", "Squamous cell carcinoma", "Atopic Dermatitis",
             "Melanocytic nevus", "Tinea Ringworm Candidiasis", "Benign keratosis", "Melanoma", "Vascular lesion"]
}

# -------------------------------- IMAGE TRANSFORM -------------------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
])

# -------------------------------- IMAGE PREDICTION FUNCTION -------------------------------- #
def predict_image(image_path, task):
    if task not in task_labels:
        raise ValueError(f"Invalid task '{task}'. Choose from: {list(task_labels.keys())}")

    # Load and process image
    image = Image.open(image_path)
    if task in ["brain", "lung"]:
        image = image.convert("L")  # Convert to grayscale

    image = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output = model(image, task)
        probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_index = torch.argmax(probs, dim=1).item()
        predicted_label = task_labels[task][predicted_index]

    return predicted_label

# -------------------------------- TEST PREDICTION -------------------------------- #
image_path = "x-rayy/pneumonia.jpeg"  # Replace with your image path
task = "lung"  # Set the task based on input type (brain, lung, or skin)

predicted_label = predict_image(image_path, task)
print(f"Predicted Label for {task}: {predicted_label}")
