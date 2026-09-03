import os, re, json, time
import requests
from bs4 import BeautifulSoup

BASE = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks"
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

struct = json.load(open(BASE + r"\_hub2_structure.json", encoding="utf-8"))
tw_urls = sorted(set(u for u, t in struct["topic_wise"]))

a2z_new = []
for line in open(BASE + r"\_overlap.txt", encoding="utf-8"):
    m = re.match(r"^\s+(https://www\.geeksforgeeks\.org/dsa/\S+) \| (.*)$", line)
    if m:
        a2z_new.append((m.group(1), m.group(2)))

EXCLUDE = {
    "https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/",  # the index itself
}
candidates = [(u, "topic-wise") for u in tw_urls] + [(u, t) for u, t in a2z_new if u not in EXCLUDE]
seen = set()
cand = []
for u, tag in candidates:
    if u not in seen:
        seen.add(u)
        cand.append((u, tag))
print("candidates to fetch:", len(cand))

SKIP_TAGS = {"script", "style", "nav", "aside", "form", "button", "iframe", "svg"}

def analyze(html):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find(class_=lambda c: c and "article--viewer_content" in c)
    if not root:
        return None
    inner = root.find(class_="content") or root
    # strip link-list containers? measure prose vs links
    text_len = len(inner.get_text(" ", strip=True))
    links = []
    for a in inner.find_all("a", href=True):
        h = a["href"].split("?")[0]
        if "/dsa/" in h:
            t = a.get_text(" ", strip=True)
            links.append((h, t))
    # prose = text of <p> elements
    ptext = sum(len(p.get_text(" ", strip=True)) for p in inner.find_all("p"))
    return {"text_len": text_len, "prose_chars": ptext, "n_links": len(links),
            "links": links}

results = {}
for i, (u, tag) in enumerate(cand, 1):
    html = fetch(u)
    if html is None:
        results[u] = {"tag": tag, "error": "fetch failed"}
        print("[%d/%d] FAIL %s" % (i, len(cand), u))
        time.sleep(2)
        continue
    info = analyze(html)
    if info is None:
        results[u] = {"tag": tag, "error": "no article container"}
        print("[%d/%d] NOCONTAINER %s" % (i, len(cand), u))
    else:
        results[u] = {"tag": tag, **info}
        print("[%d/%d] %-12s text=%-6d prose=%-6d links=%-3d %s" %
              (i, len(cand), tag, info["text_len"], info["prose_chars"], info["n_links"], u.split("/dsa/")[1][:50]))
    time.sleep(1.5)

json.dump(results, open(BASE + r"\_index_analysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved _index_analysis.json")
