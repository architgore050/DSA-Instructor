import re, json
from bs4 import BeautifulSoup

html = open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub2.html", encoding="utf-8").read()
soup = BeautifulSoup(html, "html.parser")
main = soup.find(class_=lambda c: c and "article--viewer_content" in c) or soup.body

sections = {}   # section name -> list of (url, title)
order = []
sec = None
topic_wise = []  # sub-index links under the Topic Wise h2
in_tw = False
for el in main.descendants:
    n = getattr(el, "name", None)
    if n is None:
        continue
    if n == "h2":
        t = el.get_text(" ", strip=True)
        sec = t[:80]
        order.append(("H2", sec))
        in_tw = "Topic Wise" in t
    elif n == "h3":
        sec = el.get_text(" ", strip=True)[:80]
        order.append(("H3", sec))
    elif n == "a" and el.has_attr("href"):
        h = el["href"].split("?")[0]
        t = el.get_text(" ", strip=True)
        if in_tw:
            topic_wise.append((h, t))
        elif "/dsa/" in h:
            sections.setdefault(sec, [])
            if (h, t) not in sections[sec]:
                sections[sec].append((h, t))

print("=== section link counts ===")
tot = 0
for s, links in sections.items():
    print(f"{s}: {len(links)}")
    tot += len(links)
print("TOTAL section article links:", tot)
print("\n=== Topic Wise sub-index links (%d) ===" % len(topic_wise))
for u, t in topic_wise:
    print(u, "|", t[:70])

json.dump({"sections": sections, "topic_wise": topic_wise},
          open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub2_structure.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
