"""
scripts/generate_reference_results.py
-------------------------------------
Computes reference predictions and metrics using the pinned model.
Saves:
  - demo_images/expected_predictions.csv
  - demo_images/expected_metrics.json
  - reference_results/expected_predictions.csv
  - reference_results/expected_metrics.json
"""

import os
import sys
import json
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seed import seed_everything
from src.vit_transforms import get_vit_transform
from scripts.reproduce_check import SimpleEvalDataset

def main():
    seed_everything(42)
    device = torch.device('cpu')
    model_dir = PROJECT_ROOT / 'models' / 'best_model_vit'

    print(f"[Reference Generator] Loading model from: {model_dir}")
    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    transform = get_vit_transform()

    def process_split(dataset_dir: Path, out_csv: Path, out_json: Path):
        ds = SimpleEvalDataset(dataset_dir, transform=transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        y_true, y_pred, rows = [], [], []
        with torch.no_grad():
            for images, labels, fnames, rel_paths in loader:
                images = images.to(device)
                logits = model(images).logits
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                for i in range(len(fnames)):
                    p_idx = int(preds[i].item())
                    p_label = 'real' if p_idx == 0 else 'fake'
                    fake_prob = float(probs[i][1].item())
                    auth_prob = float(probs[i][0].item())
                    t_lbl = 'real' if int(labels[i].item()) == 0 else 'fake'

                    y_true.append(int(labels[i].item()))
                    y_pred.append(p_idx)
                    rows.append({
                        'filename': fnames[i],
                        'relative_path': rel_paths[i],
                        'true_label': t_lbl,
                        'predicted_label': p_label,
                        'fake_probability': f'{fake_prob:.6f}',
                        'authentic_probability': f'{auth_prob:.6f}'
                    })

        acc = accuracy_score(y_true, y_pred)
        prec_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'relative_path', 'true_label', 'predicted_label', 'fake_probability', 'authentic_probability'])
            writer.writeheader()
            writer.writerows(rows)

        metrics = {
            'total_samples': len(y_true),
            'real_count': sum(1 for y in y_true if y == 0),
            'fake_count': sum(1 for y in y_true if y == 1),
            'accuracy': round(float(acc), 4),
            'accuracy_percent': round(float(acc) * 100, 2),
            'precision_fake': round(float(prec_fake), 4),
            'recall_fake': round(float(rec_fake), 4),
            'recall_real': round(float(rec_real), 4),
            'f1_score_fake': round(float(f1), 4),
            'confusion_matrix': {
                'true_real_pred_real': int(cm[0, 0]),
                'true_real_pred_fake': int(cm[0, 1]),
                'true_fake_pred_real': int(cm[1, 0]),
                'true_fake_pred_fake': int(cm[1, 1])
            }
        }
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"Processed {dataset_dir.name}: Acc={metrics['accuracy_percent']}%, Prec(Fake)={metrics['precision_fake']}, Rec(Real)={metrics['recall_real']}, Rec(Fake)={metrics['recall_fake']}, F1={metrics['f1_score_fake']}")
        print(f"  CM: {cm.tolist()}")
        return metrics

    m_demo = process_split(PROJECT_ROOT / 'demo_images', PROJECT_ROOT / 'demo_images' / 'expected_predictions.csv', PROJECT_ROOT / 'demo_images' / 'expected_metrics.json')
    m_test = process_split(PROJECT_ROOT / 'data' / 'small' / 'test', PROJECT_ROOT / 'reference_results' / 'expected_predictions.csv', PROJECT_ROOT / 'reference_results' / 'expected_metrics.json')

    print("\n--- FINAL METRICS SUMMARY ---")
    print("DEMO METRICS:")
    print(json.dumps(m_demo, indent=2))
    print("TEST METRICS:")
    print(json.dumps(m_test, indent=2))

if __name__ == '__main__':
    main()
