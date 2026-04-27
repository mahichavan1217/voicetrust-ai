import urllib.request, tarfile, os
from pathlib import Path

# OpenSLR resources for Marathi
urls = [
    "https://openslr.org/resources/110/marathi_female.tar.gz",
    "https://openslr.org/resources/30/marathi_newscrawl_corpus.tar.gz",
]

headers = {"User-Agent": "Mozilla/5.0"}

for url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            size = r.headers.get("content-length", "?")
            print(f"OK ({r.status}): {url}")
            print(f"  Size: {size} bytes")
    except Exception as e:
        print(f"FAIL: {url}")
        print(f"  Error: {e}")
