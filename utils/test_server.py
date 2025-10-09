# client_search.py
import requests, sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

IMG = Path("datasets/test_images/rose.jpg")
BASE = "http://127.0.0.1:8000"
SEARCH_URL = BASE + "/search"
IMAGE_URL_TEMPLATE = BASE + "/image/{}"

if not IMG.exists(): 
    print(f"Image does not exist {IMG}")
    sys.exit(1)

with IMG.open("rb") as f:
    try:
        resp = requests.post(SEARCH_URL, files={"file": (IMG.name, f, "image/jpeg")}, data={"k": 10}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e: sys.exit(1)

try: payload = resp.json()
except Exception: sys.exit(1)

results = payload.get("results") if isinstance(payload, dict) else payload
if results is None: sys.exit(1)

retrieved_images = []
for i, idx in enumerate(results):
    try:
        r = requests.get(IMAGE_URL_TEMPLATE.format(int(idx)), timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception: img = None
    retrieved_images.append((idx, img))

# Single plotting section - remove duplicate
n_results = len(retrieved_images)
fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 3))
if n_results + 1 == 1: axes = [axes]

# Show original query as first column
query_img = Image.open(IMG).convert("RGB")
axes[0].imshow(query_img)
axes[0].set_title("Query\n" + IMG.name)
axes[0].axis("off")

# Then the returned images
for i, (rec, img) in enumerate(retrieved_images, start=1):
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Image {rec}", fontsize=9)  # Simply show the image index
    ax.axis("off")

plt.tight_layout()
plt.show()