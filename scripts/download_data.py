"""Download the public Gregg shorthand datasets into data/.

    python scripts/download_data.py              # everything (default)
    python scripts/download_data.py grascii      # just some sources
    python scripts/download_data.py topaz --topaz-pages   # also page scans (~660 MB)

Sources:
  grascii   Word/phrase images from Gregg dictionaries, labelled with Grascii + English (MIT)
  gregg1916 Zhai et al. 2018 word images from the 1916 dictionary (GitHub)
  topaz     Line crops from ~24 printed Gregg books, labelled with Grascii + English.
            Pickles are converted to parquet (PNG bytes) one book at a time and then
            deleted, so peak disk use stays around one book (~400 MB).
"""

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gregg_ocr import PROCESSED, RAW, safe_pickle  # noqa: E402

GRASCII_DATASETS = [
    "grascii/gregg-anniversary-words",
    "grascii/gregg-preanniversary-words",
    "grascii/gregg-preanniversary-phrases",
]
GREGG1916_BASE = "https://raw.githubusercontent.com/anonimously/Gregg1916-Recognition/master/data/"
GREGG1916_PARTS = ["data.z01", "data.z02", "data.zip"]
TOPAZ_REPO = "amongglue/topaz-shorthand-data"
TOPAZ_PREFIX = "gregg-anniversary/processed"


def download_grascii():
    out = RAW / "grascii"
    for repo in GRASCII_DATASETS:
        name = repo.split("/")[1]
        dest = out / f"{name}.parquet"
        if dest.exists():
            print(f"  {name}: already present")
            continue
        path = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset",
                               local_dir=out / "_hf" / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, dest)
        print(f"  {name}: {len(pd.read_parquet(dest, columns=['longhand']))} rows")
    shutil.rmtree(out / "_hf", ignore_errors=True)


def download_gregg1916():
    out = RAW / "gregg1916"
    if (out / "extracted").exists():
        print("  gregg1916: already present")
        return
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for part in GREGG1916_PARTS:
        if not (parts / part).exists():
            urllib.request.urlretrieve(GREGG1916_BASE + part, parts / part)
    # The archive is split across three files; Info-ZIP's zip can rejoin it.
    merged = out / "merged.zip"
    subprocess.run(["zip", "-q", "-s", "0", str(parts / "data.zip"), "--out", str(merged)], check=True)
    dest = (out / "extracted").resolve()
    with zipfile.ZipFile(merged) as zf:
        for member in zf.namelist():
            if not (dest / member).resolve().is_relative_to(dest):
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(dest)
    merged.unlink()
    shutil.rmtree(parts)
    n = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"  gregg1916: {n} files extracted")


def _png_bytes(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _topaz_book_files(api: HfApi) -> dict[str, str]:
    """Pick one pickle per book: the `_clean` filtered set when it is non-empty, else the raw one."""
    files = {f.path: f.size for f in api.list_repo_tree(TOPAZ_REPO, f"{TOPAZ_PREFIX}/books", repo_type="dataset")}
    chosen = {}
    for path, size in files.items():
        name = path.rsplit("/", 1)[1]
        if "_before_split" in name or "_strict" in name:
            continue
        book = name.split("_v4")[0]
        if name.endswith("_clean.pkl") and size > 1000:
            chosen[book] = path
        else:
            chosen.setdefault(book, path)
    return dict(sorted(chosen.items()))


def download_topaz(with_pages: bool):
    api = HfApi()
    tmp = RAW / "topaz" / "_tmp"
    out = PROCESSED / "topaz"
    out.mkdir(parents=True, exist_ok=True)

    for sub in ["keys", "gutenberg"] + (["ocr"] if with_pages else []):
        api_files = [f.path for f in api.list_repo_tree(TOPAZ_REPO, f"{TOPAZ_PREFIX}/{sub}",
                                                        repo_type="dataset", recursive=True)
                     if hasattr(f, "size")]
        for path in tqdm(api_files, desc=f"topaz/{sub}", leave=False):
            hf_hub_download(TOPAZ_REPO, path, repo_type="dataset", local_dir=RAW / "topaz")

    for book, path in _topaz_book_files(api).items():
        dest = out / f"{book}.parquet"
        if dest.exists():
            print(f"  {book}: already present")
            continue
        local = hf_hub_download(TOPAZ_REPO, path, repo_type="dataset", local_dir=tmp)
        rows = safe_pickle.load(Path(local))
        df = pd.DataFrame([{
            "book": book,
            "page": r.get("page"),
            "para": r.get("para"),
            "grascii": r.get("label"),
            "english": r.get("words"),
            "ctc_pred": r.get("ctc"),
            "ter_ctc": r.get("ter_ctc"),
            "nll": r.get("nll"),
            "height": r["image"].shape[0],
            "width": r["image"].shape[1],
            "png": _png_bytes(r["image"]),
        } for r in tqdm(rows, desc=book, leave=False)])
        df.to_parquet(dest, index=False)
        Path(local).unlink()
        print(f"  {book}: {len(df)} lines ({path.rsplit('/', 1)[1]})")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", default=["grascii", "gregg1916", "topaz"],
                    choices=["grascii", "gregg1916", "topaz"])
    ap.add_argument("--topaz-pages", action="store_true", help="also fetch Topaz page scans (~660 MB)")
    args = ap.parse_args()

    for src in args.sources:
        print(f"[{src}]")
        if src == "grascii":
            download_grascii()
        elif src == "gregg1916":
            download_gregg1916()
        else:
            download_topaz(args.topaz_pages)


if __name__ == "__main__":
    main()
