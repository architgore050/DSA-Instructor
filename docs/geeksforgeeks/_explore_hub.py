import re, time
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
            time.sleep(3 * (a + 1))
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:100])
            time.sleep(3 * (a + 1))
    return None

html = fetch("https://www.geeksforgeeks.org/dsa/")
open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub.html", "w", encoding="utf-8").write(html)
soup = BeautifulSoup(html, "html.parser")

# find the main content container candidates
for cls in ("article--viewer_content",):
    el = soup.find(class_=lambda c: c and cls in c)
    print("container found:", bool(el))

main = soup.find(class_=lambda c: c and "article--viewer_content" in c) or soup.body
print("=== headings (h1-h4) with following link counts ===")
sec = None
for el in main.descendants:
    n = getattr(el, "name", None)
    if n is None:
        continue
    if n in ("h1", "h2", "h3"):
        print(n.upper(), "|", el.get_text(" ", strip=True)[:90])
