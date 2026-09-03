import time
import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
H = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

def fetch(url):
    for a in range(3):
        try:
            r = requests.get(url, headers=H, timeout=20)
            if r.status_code == 200:
                return r.text
            print("HTTP", r.status_code, url)
            time.sleep(4 * (a + 1))
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:100])
            time.sleep(4 * (a + 1))
    return None

url = "https://www.geeksforgeeks.org/dsa/trapping-rain-water/"
html = fetch(url)
open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_sample_article.html", "w", encoding="utf-8").write(html)
soup = BeautifulSoup(html, "html.parser")

root = soup.find(class_=lambda c: c and "article--viewer_content" in c)
inner = root.find(class_="content") if root else None
print("root found:", bool(root), "| inner .content:", bool(inner))
target = inner or root

def skel(el, depth=0, maxdepth=3):
    for ch in el.children:
        n = getattr(ch, "name", None)
        if n is None:
            t = str(ch).strip()[:60]
            if t and depth < 2:
                print("  " * depth + "TEXT:", t.replace("\n", " ")[:70])
            continue
        cls = " ".join(ch.get("class") or [])[:80]
        extra = ""
        if n == "pre":
            extra = " [PRE len=%d]" % len(ch.get_text())
        print("  " * depth + "<%s class='%s'%s>" % (n, cls, extra))
        if depth < maxdepth:
            skel(ch, depth + 1, maxdepth)

skel(target, 0, 2)
