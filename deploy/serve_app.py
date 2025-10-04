from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import numpy as np
import uvicorn
import asyncio
from typing import List, Dict, Any
from image_similarity.flower_search import FlowerImageSearch

app = FastAPI(title="Image Search Service")

# --- Singleton engine created at startup ---
ENGINE: FlowerImageSearch = None

@app.on_event("startup")
def load_engine():
    global ENGINE
    ENGINE = FlowerImageSearch(dataset='datasets/flowers102', model_location='models/kmeans_search/', batch_size=64)
    app.state.engine = ENGINE

@app.get("/health")
def health():
    return {"status": "ok"}

def read_imagefile_to_pil(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image") from e

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...), k: int = Form(10)):
    data = await file.read()
    img = read_imagefile_to_pil(data)
    # run the sync search in threadpool so event loop not blocked
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, app.state.engine.search, img, k)
    return {"results": results}

# @app.post("/search_batch")
# async def search_batch_endpoint(files: List[UploadFile] = File(...), k: int = Form(10)):
#     # read images into PIL list
#     pil_images = []
#     for f in files:
#         data = await f.read()
#         pil_images.append(read_imagefile_to_pil(data))
#     # Use engine.encode_batch and FAISS batch search for speed
#     loop = asyncio.get_event_loop()
#     results_batch = await loop.run_in_executor(None, app.state.engine.search_batch, pil_images, k)
#     # results_batch should be a list of results for each query
#     return {"results": results_batch}


@app.get("/image/{index}")
def image_by_index(index: int):
    engine = app.state.engine

    # Use index_image to reconstruct a reliable PIL image
    try:
        path, pil_img = engine.index_image(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="Index out of range")
    except Exception as e:
        # fallback: log and return 500
        raise HTTPException(status_code=500, detail=f"Could not load image: {e}")

    print('Streaming PIL image')
    # Stream the PIL image as JPEG bytes (client-friendly)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


if __name__ == "__main__":
    # dev server (use gunicorn + uvicorn worker for prod)
    uvicorn.run("serve_app:app", host="0.0.0.0", port=8000, log_level="info")
