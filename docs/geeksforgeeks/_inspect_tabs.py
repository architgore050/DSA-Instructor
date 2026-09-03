from bs4 import BeautifulSoup

html = open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_sample_article.html", encoding="utf-8").read()
soup = BeautifulSoup(html, "html.parser")
root = soup.find(class_=lambda c: c and "article--viewer_content" in c)
inner = root.find(class_="content")

tabs = inner.find("gfg-tabs")
print("=== gfg-tabs subtree (first one) ===")
def skel(el, depth=0):
    for ch in el.children:
        n = getattr(ch, "name", None)
        if n is None:
            t = str(ch).strip()
            if t:
                print("  " * depth + "TEXT:", t.replace("\n", "\\n")[:80])
            continue
        cls = " ".join(ch.get("class") or [])[:70]
        attrs = {k: v for k, v in ch.attrs.items() if k not in ("class",)}
        astr = " ".join("%s=%r" % (k, str(v)[:40]) for k, v in attrs.items())[:80]
        print("  " * depth + "<%s class='%s' %s>" % (n, cls, astr))
        if depth < 5:
            skel(ch, depth + 1)

skel(tabs)
print("\n=== code-output div ===")
co = inner.find("div", class_="code-output")
skel(co)
print("\n=== blockquote structure (snippet only) ===")
bq = inner.find("blockquote")
for el in bq.descendants:
    n = getattr(el, "name", None)
    if n is not None:
        print("<%s class='%s'>" % (n, " ".join(el.get("class") or [])[:60]))
print("snippet:", bq.get_text(" ", strip=True)[:70])
