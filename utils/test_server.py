# client_search.py
import requests
from pathlib import Path
import sys
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Config
IMG = Path("datasets/test_images/lotus.jpg")
BASE = "http://127.0.0.1:8000"
SEARCH_URL = BASE + "/search"
IMAGE_URL_TEMPLATE = BASE + "/image/{}"   # expects index

if not IMG.exists():
    print("Image not found:", IMG)
    sys.exit(1)

# 1) POST query image
with IMG.open("rb") as f:
    files = {"file": (IMG.name, f, "image/jpeg")}
    data = {"k": 10}   # top-k results
    try:
        resp = requests.post(SEARCH_URL, files=files, data=data, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Search request failed:", e)
        sys.exit(1)

print("HTTP", resp.status_code)
try:
    payload = resp.json()
except Exception:
    print("Invalid JSON response:")
    print(resp.text)
    sys.exit(1)

# Accept either {"results":[...]} or a bare list
results = payload.get("results") if isinstance(payload, dict) else payload
if results is None:
    print("No 'results' found in response:", payload)
    sys.exit(1)


# 2) Download returned images
retrieved_images = []
for i, idx in enumerate(results):
    print(f"Getting image with index: {idx}")

    img = None
    try:
        r = requests.get(IMAGE_URL_TEMPLATE.format(int(idx)), timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to fetch /image/{idx}: {e}")
        img = None

    # If still None, push a placeholder (blank)
    if img is None:
        print(f'Error fetching image!')

    retrieved_images.append((idx, img))

# 3) Plot query + results in a single row (hides axes)
n_results = len(retrieved_images)
ncols = n_results + 1  # query + results

fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))
if ncols == 1:
    axes = [axes]  # ensure iterable

# Show original query as first column
query_img = Image.open(IMG).convert("RGB")
axes[0].imshow(query_img)
axes[0].set_title("Query\n" + IMG.name)
axes[0].axis("off")

# Then the returned images
for i, (rec, img) in enumerate(retrieved_images, start=1):
    ax = axes[i]
    ax.imshow(img)
    # Title: prefer index+label, else path
    title_parts = []
    if isinstance(rec, dict):
        if "index" in rec:
            title_parts.append(f"idx:{rec['index']}")
        if "label" in rec:
            title_parts.append(f"lbl:{rec['label']}")
        if "distance" in rec:
            title_parts.append(f"d={rec['distance']:.3f}")
        if "path" in rec and rec.get("path"):
            # show only filename to keep titles short
            title_parts.append(Path(rec['path']).name)
    title = "\n".join(title_parts) if title_parts else f"res:{i-1}"
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
