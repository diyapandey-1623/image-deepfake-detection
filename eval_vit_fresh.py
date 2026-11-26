import os
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, r"c:\deepfake detection model\deepfake-detector")
from backend.model.inference import HuggingFaceModel

test_dir = r"c:\deepfake detection model\deepfake-detector\data\small\test"
real_dir = os.path.join(test_dir, "real")
fake_dir = os.path.join(test_dir, "fake")

real_files = sorted(os.listdir(real_dir))
fake_files = sorted(os.listdir(fake_dir))

model_path = r"c:\deepfake detection model\deepfake-detector\models\best_model_vit"
print(f"Loading ViT model from: {model_path}")
model = HuggingFaceModel(model_id=model_path, device="cpu")

y_true = []
y_pred = []
probs_real = []
probs_fake = []

print(f"Evaluating {len(real_files)} REAL images...")
for f in real_files:
    img_path = os.path.join(real_dir, f)
    with Image.open(img_path).convert("RGB") as img:
        res = model.predict(img)
    p = res["fake_probability"]
    y_true.append(0)
    y_pred.append(1 if p >= 0.50 else 0)
    probs_real.append(p)

print(f"Evaluating {len(fake_files)} FAKE images...")
for f in fake_files:
    img_path = os.path.join(fake_dir, f)
    with Image.open(img_path).convert("RGB") as img:
        res = model.predict(img)
    p = res["fake_probability"]
    y_true.append(1)
    y_pred.append(1 if p >= 0.50 else 0)
    probs_fake.append(p)

acc = accuracy_score(y_true, y_pred)
prec_fake = precision_score(y_true, y_pred, pos_label=1)
rec_fake = recall_score(y_true, y_pred, pos_label=1)
rec_real = recall_score(y_true, y_pred, pos_label=0)
f1_fake = f1_score(y_true, y_pred, pos_label=1)
cm = confusion_matrix(y_true, y_pred)

avg_real = float(np.mean(probs_real))
std_real = float(np.std(probs_real))
avg_fake = float(np.mean(probs_fake))
std_fake = float(np.std(probs_fake))

print("\n" + "="*60)
print("       FRESH EVALUATION RESULTS ON ALL 500 TEST IMAGES       ")
print("="*60)
print(f"Total Test Images: {len(y_true)} (250 Real, 250 Fake)")
print(f"Overall Accuracy:  {acc * 100:.2f}% ({sum(np.array(y_true) == np.array(y_pred))}/500)")
print(f"Real Recall:       {rec_real * 100:.2f}% ({cm[0,0]}/250)")
print(f"Fake Recall:       {rec_fake * 100:.2f}% ({cm[1,1]}/250)")
print(f"Fake Precision:    {prec_fake * 100:.2f}%")
print(f"Fake F1-Score:     {f1_fake * 100:.2f}%")
print("-"*60)
print(f"Avg Synthetic Probability on REAL images: {avg_real:.4f} (std: {std_real:.4f})")
print(f"Avg Synthetic Probability on FAKE images: {avg_fake:.4f} (std: {std_fake:.4f})")
print("-"*60)
print("Confusion Matrix:")
print("                 Predicted Real (0)    Predicted Fake (1)")
print(f"True Real (0)           {cm[0,0]:<21} {cm[0,1]}")
print(f"True Fake (1)           {cm[1,0]:<21} {cm[1,1]}")
print("="*60)
