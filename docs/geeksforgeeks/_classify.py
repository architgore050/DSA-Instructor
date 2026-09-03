import json, re

BASE = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks"
struct = json.load(open(BASE + r"\_hub2_structure.json", encoding="utf-8"))
ana = json.load(open(BASE + r"\_index_analysis.json", encoding="utf-8"))

sec_urls, sec_of = set(), {}
for s, links in struct["sections"].items():
    for u, t in links:
        sec_urls.add(u)
        sec_of.setdefault(u, s if s not in (None, "null") else "Step by Step Learning")

EXCLUDE = {
    "https://www.geeksforgeeks.org/dsa/geeksforgeeks-practice-best-online-coding-platform/",  # platform promo, no DSA content
}

def is_index(u):
    a = ana.get(u)
    if not a or "error" in a:
        return False
    return a["n_links"] >= 12 and a["prose_chars"] < 800

cand_urls = [u for u in ana if u not in EXCLUDE]
articles, indexes = [], []
for u in cand_urls:
    (indexes if is_index(u) else articles).append(u)

print("candidates:", len(cand_urls), "| classified articles:", len(articles), "| index pages:", len(indexes))

# harvest links from index pages -> new urls not already known
known = sec_urls | set(articles) | EXCLUDE
new_from_indexes = {}   # url -> (title, source_index_page)
for u in indexes:
    for lu, lt in ana[u]["links"]:
        if lu == u or lu in known or lu in new_from_indexes:
            continue
        new_from_indexes[lu] = (lt, u.split("/dsa/")[1].rstrip("/"))

print("\nnew urls harvested from index pages:", len(new_from_indexes))
for u in sorted(new_from_indexes):
    print("  ", u, "|", new_from_indexes[u][0][:50], "<-", new_from_indexes[u][1])

json.dump({"articles": articles, "indexes": indexes, "excluded": sorted(EXCLUDE),
           "new_from_indexes": {k: v for k, v in new_from_indexes.items()}},
          open(BASE + r"\_classify.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
