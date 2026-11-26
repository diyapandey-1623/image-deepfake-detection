import os
import requests

test_dir = r"c:\deepfake detection model\deepfake-detector\data\small\test"
real_files = sorted(os.listdir(os.path.join(test_dir, "real")))[:10]
fake_files = sorted(os.listdir(os.path.join(test_dir, "fake")))[:10]

url = "http://127.0.0.1:8000/api/analyze"

print("--- Testing 10 REAL Images ---")
for f in real_files:
    p = os.path.join(test_dir, "real", f)
    with open(p, "rb") as fp:
        r = requests.post(url, files={"file": (f, fp, "image/jpeg")})
    d = r.json()
    v = d.get("verdict")
    prob = d.get("fake_probability")
    correct = "CORRECT" if v == "real" else "WRONG"
    print(f"REAL | {f} | Pred: {v} | FakeProb: {prob:.4f} | {correct}")

print("\n--- Testing 10 FAKE Images ---")
for f in fake_files:
    p = os.path.join(test_dir, "fake", f)
    with open(p, "rb") as fp:
        r = requests.post(url, files={"file": (f, fp, "image/jpeg")})
    d = r.json()
    v = d.get("verdict")
    prob = d.get("fake_probability")
    correct = "CORRECT" if v == "fake" else "WRONG"
    print(f"FAKE | {f} | Pred: {v} | FakeProb: {prob:.4f} | {correct}")
