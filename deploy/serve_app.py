from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io, asyncio
import uvicorn
from image_similarity.flower_search import FlowerImageSearch

app = FastAPI(title="Image Search Service")
ENGINE = None

@app.on_event("startup")
def load_engine():
    global ENGINE
    ENGINE = FlowerImageSearch(dataset='datasets/flowers102', model_location='models/kmeans_search/', batch_size=64)
    app.state.engine = ENGINE

@app.get("/health")
def health(): return {"status": "ok"}

def read_imagefile_to_pil(data: bytes) -> Image.Image:
    try: return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e: raise HTTPException(status_code=400, detail="Invalid image") from e

@app.post("/search")
async def search_endpoint(file: UploadFile = File(...), k: int = Form(10)):
    img = read_imagefile_to_pil(await file.read())
    results = await asyncio.get_event_loop().run_in_executor(None, app.state.engine.search, img, k)
    return {"results": results}

@app.get("/image/{index}")
def image_by_index(index: int):
    try:
        _, pil_img = app.state.engine.index_image(index)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except IndexError: raise HTTPException(status_code=404, detail="Index out of range")
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__": 
    uvicorn.run("serve_app:app", host="0.0.0.0", port=8000, log_level="info")
