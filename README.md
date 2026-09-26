# Hugging-face-object-detection
Object-detection-that help people out
# 🚮 Trashify: Custom Object Detection with Hugging Face Transformers[cite: 1]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pradhanagnibesh-cell/Hugging-face-object-detection)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.53.0-yellow)
![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-3.6.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0%2Bcu126-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo-FF7C00)

An end-to-end computer vision project that builds and fine-tunes a custom **Object Detection** model (`RT-DETRv2`) using the **Hugging Face ecosystem**[cite: 1]. 

**Trashify** 🚮 is designed to incentivize local trash collection by detecting three core objects in an image: a **`hand`** (or **`trash_arm`**), **`trash`**, and a **`bin`**[cite: 1]. When all three target objects are detected together, the user scores **+1 point**[cite: 1]!

---

## 📌 Project Workflow (*Data, Model, Demo!*)[cite: 1]

This repository follows a reproducible end-to-end machine learning workflow using Hugging Face tools[cite: 1]:

1. **📦 Data Preparation (`datasets`):** Load and explore the manually collected and Prodigy-annotated [`mrdbourke/trashify_manual_labelled_images`](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images) dataset from the Hugging Face Hub (`1,128` images)[cite: 1].
2. **🎨 Bounding Box Processing & Visualization (`torchvision` & `matplotlib`):** Convert bounding box coordinates between `XYWH`, `XYXY`, and `CXCYWH` formats and render custom color-coded bounding boxes[cite: 1].
3. **🧠 Model Fine-Tuning (`transformers`):** Load a pre-trained object detection architecture (`AutoModelForObjectDetection` / `RT-DETRv2`), configure `TrainingArguments`, and fine-tune it on the custom dataset using the Hugging Face `Trainer` API[cite: 1].
4. **📈 Evaluation (`torchmetrics` & `pycocotools`):** Evaluate object detection performance using COCO metrics on test predictions[cite: 1].
5. **🚀 Interactive Demo (`gradio` & Hugging Face Spaces):** Package the fine-tuned model into a shareable web application[cite: 1].

---

## 🏷️ Dataset & Target Classes[cite: 1]

The dataset is loaded directly from the Hugging Face Hub (`mrdbourke/trashify_manual_labelled_images`) and contains `1,128` manually photographed and labeled images across **7 categories** (4 positive classes and 3 hard-negative classes)[cite: 1]:

| Class ID | Class Label | Type | Description | Bounding Box Color (RGB) |
| :---: | :--- | :---: | :--- | :--- |
| `0` | `bin` | ✅ Positive | A rubbish bin or trash can | 🔵 `(0, 0, 224)` Bright Blue |
| `1` | `hand` | ✅ Positive | A person's hand | 🟣 `(148, 0, 211)` Dark Purple |
| `2` | `not_bin` | ❌ Negative | Items that look like a bin but shouldn't be classified as one | 🔴 `(255, 80, 80)` Light Red |
| `3` | `not_hand` | ❌ Negative | Items that look like a hand | 🔴 `(255, 80, 80)` Light Red |
| `4` | `not_trash` | ❌ Negative | Items that look like trash | 🔴 `(255, 80, 80)` Light Red |
| `5` | `trash` | ✅ Positive | Litter (plastic bottles, wrappers, coffee cups, cigarette butts) | 🟢 `(0, 255, 0)` Bright Green |
| `6` | `trash_arm` | ✅ Positive | A mechanical grabber arm used for picking up trash | 🟠 `(255, 140, 0)` Deep Orange |

---

## 🛠️ Tech Stack & Requirements[cite: 1]

* **Deep Learning & Vision:** `torch` (`2.7.0+cu126`), `torchvision`, `transformers` (`4.53.0`)[cite: 1]
* **Dataset & Metrics:** `datasets` (`3.6.0`), `torchmetrics[detection]` (`1.7.1`), `pycocotools`, `evaluate`, `accelerate`[cite: 1]
* **Image Processing & Plotting:** `Pillow (PIL)`, `matplotlib`, `numpy`[cite: 1]
* **Deployment / UI:** `gradio`[cite: 1]

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/pradhanagnibesh-cell/Hugging-face-object-detection.git](https://github.com/pradhanagnibesh-cell/Hugging-face-object-detection.git)
cd Hugging-face-object-detection
```

### 2. Install Dependencies[cite: 1]
```bash
pip install -U transformers datasets gradio accelerate evaluate
pip install -U torch torchvision "torchmetrics[detection]" pycocotools pillow matplotlib numpy
```

### 3. Run the Jupyter Notebook[cite: 1]
Open the `.ipynb` notebook in **Jupyter Lab**, **VS Code**, or **Google Colab**[cite: 1]:
> **💡 Note:** If you are running the notebook in Google Colab, make sure to enable a GPU by navigating to **`Runtime` ➡️ `Change runtime type` ➡️ `Hardware accelerator` ➡️ `GPU`**[cite: 1].

---

## 💻 Quick Code Preview: Loading & Visualizing Annotations[cite: 1]

Here is a snippet from the notebook showing how the dataset is loaded from the Hugging Face Hub and how `XYWH` annotations are converted to `XYXY` for visualization with `torchvision`[cite: 1]:

```python
import torch
import torchvision
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes

# 1. Load the Trashify dataset from Hugging Face Hub
dataset = load_dataset(path="mrdbourke/trashify_manual_labelled_images")

# 2. Extract class names and mappings
categories = dataset["train"].features["annotations"]["category_id"]
id2label = {i: name for i, name in enumerate(categories.feature.names)}

# 3. Inspect a sample and convert bounding boxes from XYWH -> XYXY
sample = dataset["train"][42]
image = sample["image"]
boxes_xywh = torch.tensor(sample["annotations"]["bbox"])
boxes_xyxy = torchvision.ops.box_convert(boxes_xywh, in_fmt="xywh", out_fmt="xyxy")
labels = [id2label[cat_id] for cat_id in sample["annotations"]["category_id"]]

# 4. Draw bounding boxes on the image tensor
image_tensor = pil_to_tensor(image)
annotated_tensor = draw_bounding_boxes(image_tensor, boxes=boxes_xyxy, labels=labels, width=3)
annotated_image = to_pil_image(annotated_tensor)
annotated_image.show()
```

---

## 🙏 Acknowledgements[cite: 1]

* Based on the [Learn Hugging Face Object Detection Tutorial](https://www.learnhuggingface.com/notebooks/hugging_face_object_detection_tutorial) and dataset by **Daniel Bourke (@mrdbourke)**[cite: 1].

## 👤 Author

**Ayushman Pradhan**
* GitHub: [@pradhanagnibesh-cell](https://github.com/pradhanagnibesh-cell)
