from pathlib import Path
import shutil

SRC_DIR = Path("data/frames/beeVideo") # Change the video name here
DST_DIR = Path("data/frames_subset/beeVideo") # Change the video name here
DST_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 250

def main():
    imgs = sorted(SRC_DIR.glob("*.jpg"))
    if not imgs:
        raise RuntimeError(f"No jpg files found in {SRC_DIR}")

    step = max(1, len(imgs) // TARGET)
    picked = imgs[::step][:TARGET]

    for p in picked:
        shutil.copy2(p, DST_DIR / p.name)

    print(f"Total in source: {len(imgs)}")
    print(f"Copied: {len(picked)} to {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()
