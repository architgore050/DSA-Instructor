import re
from bs4 import BeautifulSoup

html = open(r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_hub.html", encoding="utf-8").read()
soup = BeautifulSoup(html, "html.parser")
main = soup.find(class_=lambda c: c and "article--viewer_content" in c)

CORE_SECTIONS = {
    "Fundamentals",            # complexity analysis, big-O, time/space
    "Array & String",          # core DS + classic problems
    "Searching",               # linear/binary search family
    "Sorting",                 # sorting algorithms
    "Bit Manipulation",        # classic topic
    "Hashing",                 # core DS
    "Backtracking",            # algorithm paradigm
    "Linked-list",             # core DS
    "Stack",                   # core DS
    "Queue",                   # core DS
    "Deque",                   # core DS
    "Binary Tree",             # core DS
    "Binary Search Tree",      # core DS
    "Heap",                    # core DS
    "Graph",                   # core DS + algorithms
    "Greedy",                  # algorithm paradigm
    "Dynamic Programming",     # algorithm paradigm
}
CHERRY_PICK_TITLES = {"Recursion", "Analysis of Recursion"}  # from Maths section

urls, seen = [], set()
sec = None
skipped_sections = {}
for el in main.descendants:
    name = getattr(el, "name", None)
    if name is None:
        continue
    if name == "h2":
        sec = el.get_text(" ", strip=True)[:80]
    elif name == "h3":
        sec = el.get_text(" ", strip=True)[:80]
    elif name == "a" and el.has_attr("href"):
        h = el["href"].split("?")[0]
        if "/dsa/" not in h or h in seen:
            continue
        t = el.get_text(" ", strip=True)
        is_core = sec in CORE_SECTIONS
        is_pick = (sec == "Maths, Pattern & Recursion" and t in CHERRY_PICK_TITLES)
        if is_core or is_pick:
            seen.add(h)
            urls.append(h)
        else:
            skipped_sections[sec] = skipped_sections.get(sec, 0) + 1

out = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks\_urls.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(urls) + "\n")

print("core urls:", len(urls))
print("skipped by section:")
for k in sorted(skipped_sections):
    print(f"  {k}: {skipped_sections[k]}")
print("total skipped:", sum(skipped_sections.values()))
