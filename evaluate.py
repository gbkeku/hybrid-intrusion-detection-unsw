import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

def load_json(name: str):
    path = MODELS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None

def main():
    files = ["ml_metrics.json", "cnn_metrics.json", "autoencoder_metrics.json", "hybrid_metrics.json"]
    for name in files:
        payload = load_json(name)
        if payload is None:
            print(f"{name}: not found")
            continue
        print(f"\n=== {name} ===")
        print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
