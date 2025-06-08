import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/tooth-agenesis-siglip2"  # Update with actual model name on Hugging Face
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Calculus",
    "1": "Caries",
    "2": "Gingivitis",
    "3": "Mouth Ulcer",
    "4": "Tooth Discoloration",
    "5": "hypodontia"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=6, label="Dental Condition Classification"),
    title="Tooth Agenesis Detection",
    description="Upload a dental image to detect conditions such as Calculus, Caries, Gingivitis, Mouth Ulcer, Tooth Discoloration, or Hypodontia."
)

if __name__ == "__main__":
    iface.launch()
