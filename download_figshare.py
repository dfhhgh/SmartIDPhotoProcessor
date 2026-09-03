"""Download Face Research Lab London Set from Figshare."""
import requests
import os
import sys
import time


def download_figshare():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    # Step 1: Visit article page to get cookies
    print("Step 1: Visiting article page...")
    try:
        resp = session.get(
            "https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666",
            timeout=30,
        )
        print(f"  Status: {resp.status_code}")
        print(f"  Cookies: {dict(session.cookies)}")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 2: Try download
    print("\nStep 2: Attempting download...")
    dest = "datasets/non_celebrity-v1/raw/neutral_front.zip"
    try:
        resp = session.get(
            "https://figshare.com/ndownloader/files/8541961",
            timeout=60,
            allow_redirects=True,
        )
        ct = resp.headers.get("content-type", "unknown")
        print(f"  Status: {resp.status_code}")
        print(f"  Content-Type: {ct}")
        print(f"  Content-Length: {len(resp.content)}")

        if len(resp.content) > 1000:
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"  Saved: {len(resp.content)} bytes")
        else:
            print(f"  Response preview: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    download_figshare()
