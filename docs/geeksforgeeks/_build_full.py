import json

BASE = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks"
struct = json.load(open(BASE + r"\_hub2_structure.json", encoding="utf-8"))
c = json.load(open(BASE + r"\_classify.json", encoding="utf-8"))

EXCLUDE = set(c["excluded"]) | {
    "https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/",  # the index page itself
}

sections = {}   # section -> [urls] in page order
order = []
for s, links in struct["sections"].items():
    name = "Step by Step Learning" if s in (None, "null") else s
    for u, t in links:
        if u in EXCLUDE:
            continue
        sections.setdefault(name, [])
        if u not in sections[name]:
            sections[name].append(u)

# topic-wise pages that are real articles -> their own section
tw_articles = [u for u in c["articles"]
               if any(u == tw for tw, _ in struct["topic_wise"]) and u not in EXCLUDE]
sections.setdefault("Topic-Wise Tutorials", [])
for u in tw_articles:
    if u not in sections["Topic-Wise Tutorials"]:
        sections["Topic-Wise Tutorials"].append(u)

# A2Z-sourced articles (candidates that are not topic-wise and not already placed)
placed = set(u for v in sections.values() for u in v)
a2z_articles = [u for u in c["articles"] if u not in tw_articles and u not in EXCLUDE]
sections.setdefault("A2Z Reference", [])
for u in a2z_articles:
    if u not in placed:
        sections["A2Z Reference"].append(u)

# harvested urls from index pages, attributed to source index topic
idx_title = {u.split("/dsa/")[1].rstrip("/"): t for u, t in struct["topic_wise"]}
a2z_titles = {}
for line in open(BASE + r"\_overlap.txt", encoding="utf-8"):
    import re as _re
    m = _re.match(r"^\s+(https://www\.geeksforgeeks\.org/dsa/\S+) \| (.*)$", line)
    if m:
        a2z_titles[m.group(1)] = m.group(2).strip()

index_set = set(c["indexes"])  # pure link-list pages: never save as articles
harvest_sections = {}
for u, (title, src_slug) in c["new_from_indexes"].items():
    if u in EXCLUDE or u in placed or u in index_set:
        continue
    src_title = idx_title.get(src_slug) or a2z_titles.get("https://www.geeksforgeeks.org/dsa/" + src_slug + "/", src_slug)
    secname = "%s (index harvest)" % src_title
    harvest_sections.setdefault(secname, [])
    if u not in harvest_sections[secname]:
        harvest_sections[secname].append(u)

all_urls = []
seen = set()
final_sections = {}
for name in list(sections):
    urls = [u for u in sections[name] if u not in seen and u not in EXCLUDE]
    for u in urls:
        seen.add(u)
    final_sections[name] = urls
    all_urls.extend(urls)

for name in sorted(harvest_sections):
    urls = [u for u in harvest_sections[name] if u not in seen and u not in EXCLUDE]
    for u in urls:
        seen.add(u)
    final_sections[name] = urls
    all_urls.extend(urls)

with open(BASE + r"\_urls_full.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_urls) + "\n")

meta = {
    "sections": final_sections,
    "excluded_non_articles": sorted(EXCLUDE),
    "index_pages_not_saved": c["indexes"],
    "total_urls": len(all_urls),
}
json.dump(meta, open(BASE + r"\_sections.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("TOTAL urls:", len(all_urls))
for name in final_sections:
    print(f"  {name}: {len(final_sections[name])}")
