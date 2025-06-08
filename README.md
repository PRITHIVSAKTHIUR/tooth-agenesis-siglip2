![123.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/9_mjg7bzNcQT-Ifx-ATmH.png)

# tooth-agenesis-siglip2

> tooth-agenesis-siglip2 is a vision-language encoder model fine-tuned from `google/siglip2-base-patch16-512` for **multi-class image classification**. It is trained to detect various **dental anomalies and conditions** such as **Calculus**, **Caries**, **Gingivitis**, **Mouth Ulcer**, **Tooth Discoloration**, and **Hypodontia**. The model uses the `SiglipForImageClassification` architecture.

> \[!note]
> SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
> [https://arxiv.org/pdf/2502.14786](https://arxiv.org/pdf/2502.14786)

```py
Classification Report:
                     precision    recall  f1-score   support

           Calculus     0.6640    0.7623    0.7098      1296
             Caries     0.9525    0.9558    0.9541      2601
         Gingivitis     0.8496    0.7842    0.8156      2349
        Mouth Ulcer     0.9939    0.9893    0.9916      2806
Tooth Discoloration     0.9314    0.9757    0.9530      2017
         hypodontia     0.9983    0.9161    0.9554      1251

           accuracy                         0.9096     12320
          macro avg     0.8983    0.8972    0.8966     12320
       weighted avg     0.9132    0.9096    0.9105     12320
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/vCoLKevXThpp6GhYCvoCe.png)

---

## Label Space: 6 Classes

```
Class 0: Calculus  
Class 1: Caries  
Class 2: Gingivitis  
Class 3: Mouth Ulcer  
Class 4: Tooth Discoloration  
Class 5: hypodontia
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
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
```

---

## Intended Use

`tooth-agenesis-siglip2` is designed for:

* **Dental Diagnosis Support** – Assists dentists and clinicians in identifying common dental conditions from images.
* **Oral Health Monitoring** – A tool for regular monitoring of dental health in clinical or remote settings.
* **Tele-dentistry** – Enables automated screening in virtual consultations and rural healthcare setups.
* **Research and Education** – Useful for academic institutions and training platforms for demonstrating AI in dental diagnostics.
* **Early Detection** – Helps identify oral health issues early to prevent progression. 
