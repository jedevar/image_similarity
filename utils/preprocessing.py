from pathlib import Path
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

IMAGE_EXTS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def find_image_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS]

def check_image(path: Path) -> Tuple[bool, str]:
    try:
        if path.stat().st_size == 0:
            return False, "zero-size file"
        with Image.open(path) as img:
            img.verify()
        return True, "Unknown error"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def scan_folder_parallel(root_folder: str, quarantine_folder: str, max_workers: int = 4) -> List[Tuple[Path, str]]:
    root = Path(root_folder)
    files: List[Path] = find_image_files(root)
    bad: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = ex.map(check_image, files)
        for p, (ok, err) in zip(files, results):
            if not ok:
                bad.append((p, err))
                print(f"[BAD] {p} -> {err}")
                if quarantine_folder:
                    q = Path(quarantine_folder)
                    q.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(q / p.name))
    print(f"Scanned {len(files)} images — found {len(bad)} bad files.")
    return bad
