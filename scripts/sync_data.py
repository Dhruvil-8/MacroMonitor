import shutil
import os

src = os.path.join("web", "public", "data", "v1")
dst = os.path.join("web", "out", "data", "v1")
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"Copied {src} to {dst} successfully.")
