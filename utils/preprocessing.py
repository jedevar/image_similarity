from pathlib import Path
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def find_image_files(root: Path): return [p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS]

def check_image(path: Path):
    try:
        if path.stat().st_size == 0: return False, "zero-size file"
        with Image.open(path) as img: img.verify()
        return True, "Unknown error"
    except Exception as e: return False, f"{type(e).__name__}: {e}"

def scan_folder_parallel(root_folder: str, quarantine_folder: str, max_workers: int = 4):
    files, bad = find_image_files(Path(root_folder)), []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p, (ok, err) in zip(files, ex.map(check_image, files)):
            if not ok:
                bad.append((p, err))
                print(f"[BAD] {p} -> {err}")
                if quarantine_folder:
                    q = Path(quarantine_folder)
                    q.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(q / p.name))
    print(f"Scanned {len(files)} images — found {len(bad)} bad files.")
    return bad
