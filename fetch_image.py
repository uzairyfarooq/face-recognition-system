import os
import re
import sys
import json
import base64
import time
import hashlib
import argparse
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import cv2

# ---------------- .env LOADER ----------------
def load_env():
    """Read the .env file (next to this script or in the working directory) into os.environ.
    The .env file takes precedence over pre-existing system environment variables."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                os.environ[key] = val
        return path
    return None

load_env()

# ---------------- CONFIG & ENVIRONMENT SETUP ----------------
API_URL    = os.getenv("API_URL", "")
API_USER   = os.getenv("API_USER", "")      
API_PASS   = os.getenv("API_PASS", "")
API_TOKEN  = os.getenv("API_TOKEN", "")     
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() in ("true", "1", "yes")

KNOWN_DIR  = "known_faces"
MANIFEST   = os.path.join(KNOWN_DIR, "_manifest.json")   # Maps CNIC/ID to folder names
WORKERS    = 8

NAME_FIELD   = None        # Optional explicit configuration e.g. "emp_name"
CNIC_FIELD   = None        # Optional explicit configuration e.g. "cnic"
IMAGE_FIELDS = None        # Optional explicit configuration e.g. ["photo", "photo2"]

NAME_HINTS  = ["full_name", "fullname", "emp_name", "employee_name", "name", "patient_name"]
IMAGE_HINTS = ["image", "img", "photo", "pic", "picture", "face", "snap"]

# Setup requests session headers and authentication from environment variables
session = requests.Session()
session.verify = VERIFY_SSL
session.headers["User-Agent"] = "Mozilla/5.0"
session.headers["Accept"] = "application/json"

if API_TOKEN:
    session.headers["Authorization"] = f"Bearer {API_TOKEN}"
elif API_USER:
    session.auth = (API_USER, API_PASS)

if not VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings()


# ---------------- FETCH ALL RECORDS ----------------
def extract_items(data):
    """Extract list of records from different REST response formats (list, dict with items, etc.)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return data["items"]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def get_json(url, **kw):
    """Perform an HTTP GET request and return parsed JSON data."""
    r = session.get(url, timeout=180, **kw)
    r.raise_for_status()
    return r.json()


def fetch_all():
    """Generator that handles pagination (hasMore/links/offset) to fetch all records from the API."""
    data = get_json(API_URL)
    total = 0
    items = extract_items(data)
    total += len(items)
    for it in items:
        yield it

    while isinstance(data, dict) and data.get("hasMore"):
        nxt = next((l.get("href") for l in data.get("links", []) if l.get("rel") == "next"), None)
        try:
            data = get_json(nxt) if nxt else get_json(API_URL, params={"offset": total})
        except Exception as e:
            print(f"\n[WARN] Next page fetch failed: {e}")
            break
        items = extract_items(data)
        if not items:
            break
        total += len(items)
        for it in items:
            yield it
        print(f"  fetched {total} records ...", end="\r")


# ---------------- FIELD DETECTION HELPERS ----------------
def find_name(rec):
    """Automatically detect person name from record dictionary using hint fields."""
    if NAME_FIELD:
        v = rec.get(NAME_FIELD)
        return v.strip() if isinstance(v, str) and v.strip() else None
    lm = {k.lower(): k for k in rec}
    for h in NAME_HINTS:
        if h in lm and isinstance(rec[lm[h]], str) and rec[lm[h]].strip():
            return rec[lm[h]].strip()
    for k in lm:
        if "name" in k and isinstance(rec[lm[k]], str) and rec[lm[k]].strip():
            return rec[lm[k]].strip()
    return None


def find_cnic(rec):
    """Automatically detect CNIC or unique identifier from record dictionary."""
    if CNIC_FIELD:
        return str(rec.get(CNIC_FIELD) or "")
    for k, v in rec.items():
        if "cnic" in k.lower() or k.lower() in ("nic", "nic_no", "id_card"):
            return str(v or "")
    return ""


def image_values(rec):
    """Extract potential image fields or values from the record dictionary."""
    if IMAGE_FIELDS:
        return [rec.get(f) for f in IMAGE_FIELDS]
    vals = [v for k, v in rec.items() if any(h in k.lower() for h in IMAGE_HINTS)]
    return vals if vals else list(rec.values())     


B64_RE = re.compile(r"^[A-Za-z0-9+/=\r\n_-]+$")


def collect_images(v, out):
    """Recursively extract image URLs or base64 strings from nested lists, dictionaries, or strings."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("data:image"):
            out.append(("b64", s.split(",", 1)[1]))
        elif s.lower().startswith("http") or re.search(r"\.(jpe?g|png|bmp|webp)(\?.*)?$", s, re.I):
            out.append(("url", urllib.parse.urljoin(API_URL, s)))
        elif len(s) > 500 and B64_RE.match(s):
            out.append(("b64", s))
    elif isinstance(v, list):
        for x in v:
            collect_images(x, out)
    elif isinstance(v, dict):
        for x in v.values():
            collect_images(x, out)


def get_bytes(kind, val):
    """Download image from URL or decode base64 string into raw bytes."""
    try:
        if kind == "url":
            r = session.get(val, timeout=60)
            r.raise_for_status()
            return r.content
        return base64.b64decode(val + "=" * (-len(val) % 4))
    except Exception as e:
        print(f"\n  [WARN] Image download/decode failed: {e}")
        return None


def valid_image(b):
    """Verify raw bytes can be decoded into a valid OpenCV image matrix."""
    if not b:
        return None
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


# ---------------- FILE & FOLDER UTILITIES ----------------
def safe_folder(name):
    """Sanitize string to create a safe directory name."""
    name = re.sub(r'[\\/:*?"<>|]+', "", name).strip().strip(".")
    return re.sub(r"\s+", " ", name) or "unknown"


def process(rec, manifest):
    """Process a single API record, download its reference images, and save them to disk."""
    name = find_name(rec)
    if not name:
        return 0
    key = find_cnic(rec) or name
    folder = manifest.get(key)
    if not folder:
        return 0

    refs = []
    for v in image_values(rec):
        collect_images(v, refs)
    if not refs:
        return 0

    out_dir = os.path.join(KNOWN_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for kind, val in refs:
        b = get_bytes(kind, val)
        img = valid_image(b)
        if img is None:
            continue
        fn = hashlib.sha1(b).hexdigest()[:12] + ".jpg"   
        path = os.path.join(out_dir, fn)
        if not os.path.exists(path):
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                tmp = path + ".part"                 
                with open(tmp, "wb") as fh:          
                    fh.write(buf.tobytes())
                os.replace(tmp, path)
                saved += 1
    if os.path.isdir(out_dir) and not os.listdir(out_dir):
        os.rmdir(out_dir)
    return saved


# ---------------- EXECUTION MODES ----------------
def inspect_mode():
    """Inspect API structure and print preview of JSON response."""
    r = session.get(API_URL, timeout=180)
    print("HTTP Status:", r.status_code, "| Content-Type:", r.headers.get("content-type"))
    try:
        data = r.json()
    except Exception:
        print(r.text[:1000])
        return

    def short(x):
        if isinstance(x, dict):
            return {k: short(v) for k, v in x.items()}
        if isinstance(x, list):
            return [short(x[0]), f"... ({len(x)} items)"] if x else []
        if isinstance(x, str) and len(x) > 80:
            return x[:60] + f"... [len={len(x)}]"
        return x

    print(json.dumps(short(data), indent=2, ensure_ascii=False))


def download_mode():
    """Main synchronization function to fetch records and download face images concurrently."""
    os.makedirs(KNOWN_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)

    records = list(fetch_all())
    print(f"\nTotal records fetched: {len(records)}")
    if records:
        print("Available fields in records:", list(records[0].keys()))

    for rec in records:
        name = find_name(rec)
        if not name:
            continue
        cnic = find_cnic(rec)
        key = cnic or name
        if key not in manifest:
            folder = safe_folder(name)
            if folder in set(manifest.values()):
                folder = f"{folder}_{cnic[-4:] if cnic else len(manifest)}"
            manifest[key] = folder

    total_imgs, people = 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, n in enumerate(ex.map(lambda r: process(r, manifest), records), 1):
            total_imgs += n
            people += 1 if n else 0
            if i % 25 == 0 or i == len(records):
                print(f"  {i}/{len(records)} records processed | {total_imgs} images saved", end="\r")

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\nSynchronization complete. {people} people, {total_imgs} new images saved in '{KNOWN_DIR}/'.")


def watch_mode(minutes):
    """Run synchronization continuously at specified minute intervals."""
    print(f"Watch mode active: checking for updates every {minutes} minute(s). Press Ctrl+C to stop.")
    while True:
        try:
            download_mode()
        except KeyboardInterrupt:
            raise
        except Exception as e:                   
            print(f"\n[ERROR] {type(e).__name__}: {e}")
        print(f"Next check in {minutes} minute(s) at ({time.strftime('%H:%M:%S')}) ...")
        time.sleep(minutes * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sync user profile face images from API to local storage.")
    ap.add_argument("--inspect", action="store_true", help="Inspect API payload structure")
    ap.add_argument("--watch", action="store_true", help="Run continuously in watch/polling mode")
    ap.add_argument("--interval", type=float, default=10, help="Interval in minutes for watch mode (default: 10)")
    args = ap.parse_args()
    try:
        if args.inspect:
            inspect_mode()
        elif args.watch:
            watch_mode(args.interval)
        else:
            download_mode()
    except KeyboardInterrupt:
        print("\nProcess terminated by user.")