import json, re

base = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks"
struct = json.load(open(base + r"\_hub2_structure.json", encoding="utf-8"))

sec_urls = set()
for s, links in struct["sections"].items():
    for u, t in links:
        sec_urls.add(u)
tw_urls = [u for u, t in struct["topic_wise"]]
tw_set = set(tw_urls)
print("section urls:", len(sec_urls), "| topic-wise unique:", len(tw_set))

# A2Z hub links from _hub_links.txt (url | title lines after the header lines)
a2z = {}
for line in open(base + r"\_hub_links.txt", encoding="utf-8"):
    m = re.match(r"^(https://www\.geeksforgeeks\.org/\S+) \| (.*)$", line.strip())
    if m:
        a2z[m.group(1)] = m.group(2)
print("a2z links:", len(a2z))

new_from_a2z = sorted(set(a2z) - sec_urls - tw_set)
print("\nA2Z urls NOT in sections or topic-wise (%d):" % len(new_from_a2z))
for u in new_from_a2z:
    print(" ", u, "|", a2z[u][:60])

overlap_tw = sorted(tw_set & set(a2z))
print("\nA2Z urls that ARE topic-wise pages (%d):" % len(overlap_tw))
for u in overlap_tw:
    print(" ", u)

# what are the 'None' section links?
print("\nNone-section links:")
for k in ("None", "null"):
    for u, t in struct["sections"].get(k, []):
        print(" ", u, "|", t[:60])
