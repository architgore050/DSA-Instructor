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
            time.sleep(3 * (a + 1))
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:100])
            time.sleep(3 * (a + 1))
    return None

url = "https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/"
html = fetch(url)
open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub2.html", "w", encoding="utf-8").write(html)
soup = BeautifulSoup(html, "html.parser")

main = soup.find(class_=lambda c: c and "article--viewer_content" in c) or soup.body
print("container:", bool(main))
import re as _re
for h in main.find_all(_re.compile("^h[1-4]$")):
    print(h.name, "|", h.get_text(" ", strip=True)[:90])
