import re
from bs4 import BeautifulSoup

html = open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub.html", encoding="utf-8").read()
print("len:", len(html))
soup = BeautifulSoup(html, "html.parser")

# all headings anywhere on page
hs = soup.find_all(re.compile("^h[1-4]$"))
print("total h1-h4 on page:", len(hs))
for h in hs[:60]:
    print(h.name, "|", h.get_text(" ", strip=True)[:90])

# links to /dsa/ or dsa-tutorial
links = {}
for a in soup.find_all("a", href=True):
    h = a["href"].split("?")[0]
    if "/dsa/" in h or "dsa-tutorial" in h:
        t = a.get_text(" ", strip=True)[:60]
        links.setdefault(h, t)
print("\ntotal distinct /dsa/ + dsa-tutorial links:", len(links))
for u in sorted(links):
    print(u, "|", links[u])
