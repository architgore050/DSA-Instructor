from bs4 import BeautifulSoup

html = open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_sample_article.html", encoding="utf-8").read()
soup = BeautifulSoup(html, "html.parser")
root = soup.find(class_=lambda c: c and "article--viewer_content" in c)

print("=== children of article--viewer_content ===")
for ch in root.children:
    n = getattr(ch, "name", None)
    if n is None:
        continue
    cls = " ".join(ch.get("class") or [])[:80]
    print("<%s class='%s'>" % (n, cls))

inner = root.find(class_="content")
print("\n=== children of .content ===")
for ch in inner.children:
    n = getattr(ch, "name", None)
    if n is None:
        continue
    cls = " ".join(ch.get("class") or [])[:80]
    print("<%s class='%s'>" % (n, cls))

text = inner.find(class_="text")
print("\n=== last 6 children of .text ===")
kids = [c for c in text.children if getattr(c, "name", None) is not None]
for ch in kids[-6:]:
    cls = " ".join(ch.get("class") or [])[:80]
    t = ch.get_text(" ", strip=True)[:70]
    print("<%s class='%s'> text=%r" % (ch.name, cls, t))

# h1 + meta
h1 = soup.find("h1")
print("\nh1:", h1.get_text(" ", strip=True) if h1 else None)
og = soup.find("meta", attrs={"property": "og:title"})
print("og:title:", og["content"][:80] if og and og.has_attr("content") else None)

# check for 'Suggested' / related blocks anywhere in root
for el in root.find_all(class_=lambda c: c and any(k in c.lower() for k in ("suggest", "related", "further"))):
    print("RELATED-BLOCK:", el.name, " ".join(el.get("class") or [])[:80], "|", el.get_text(" ", strip=True)[:60])
