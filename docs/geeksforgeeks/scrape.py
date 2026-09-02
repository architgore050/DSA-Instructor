import os, re, sys, time, random, argparse
import requests
from bs4 import BeautifulSoup

BASE = r"C:\Users\gorea\OneDrive\Desktop\DSA Instructor\docs\geeksforgeeks"
URLS_FILE = os.path.join(BASE, "_urls.txt")
FAILURES_FILE = os.path.join(BASE, "_failures.txt")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SKIP_TAGS = {"script", "style", "nav", "aside", "form", "button", "iframe",
             "svg", "img", "gfg-carousel", "gfg-carousel-content"}
BAD_CLASS_RE = re.compile(r"banner|ad-|^ad$|carousel|related|author|comment|share|social|newsletter", re.I)


def slugify(url):
    seg = [s for s in url.split("/") if s]
    last = seg[-1] if seg else "index"
    s = re.sub(r"[^A-Za-z0-9_\-]", "-", last)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s or "index").lower()


def render(el, out):
    name = el.name
    if name is None:
        return
    if name in SKIP_TAGS:
        return
    cls = " ".join(el.get("class") or [])
    if BAD_CLASS_RE.search(cls):
        return
    if name == "pre":
        lines = [l.rstrip() for l in el.get_text().splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            pads = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
            pad = min(pads) if pads else 0
            out.append("")
            for l in lines:
                out.append(l[pad:] if len(l) >= pad else l)
            out.append("")
        return
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        t = el.get_text(" ", strip=True)
        if t:
            out.append("")
            out.append("#" * int(name[1]) + " " + t)
            out.append("")
        return
    if name == "p":
        t = el.get_text(" ", strip=True)
        if t:
            out.append(t)
            out.append("")
        return
    if name in ("ul", "ol"):
        for i, li in enumerate(el.find_all("li", recursive=False), 1):
            t = li.get_text(" ", strip=True)
            if t:
                out.append(("%d. " % i if name == "ol" else "- ") + t)
        out.append("")
        return
    if name == "blockquote":
        t = el.get_text(" ", strip=True)
        if t:
            out.append("> " + t)
            out.append("")
        return
    if name == "table":
        for tr in el.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            if any(cells):
                out.append(" | ".join(cells))
        out.append("")
        return
    for ch in el.children:
        render(ch, out)


def extract(html):
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("h1")
    if not title_el:
        og = soup.find("meta", attrs={"property": "og:title"})
        title = og["content"].split(" - GeeksforGeeks")[0] if og else "(untitled)"
    else:
        title = title_el.get_text(" ", strip=True)

    root = None
    for cls in ("article--viewer_content",):
        root = soup.find(class_=lambda c: c and cls in c)
        if root:
            break
    if not root:
        root = soup.find("main") or soup.body or soup
    inner = root.find(class_="content")
    if inner:
        root = inner

    out = []
    render(root, out)
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    return title, body


def fetch(url):
    last_err = None
    for attempt in range(3):  # initial + 2 retries
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text, None
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = "HTTP %d" % r.status_code
                time.sleep(3 * (attempt + 1))
                continue
            return None, "HTTP %d" % r.status_code
        except requests.exceptions.RequestException as e:
            last_err = "%s: %s" % (type(e).__name__, str(e)[:120])
            time.sleep(3 * (attempt + 1))
    return None, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    urls = [u.strip() for u in open(URLS_FILE, encoding="utf-8") if u.strip()]
    if args.limit:
        urls = urls[:args.limit]

    saved = skipped = failed = 0
    consec_blocks = 0
    used_slugs = set(f[:-4] for f in os.listdir(BASE) if f.endswith(".txt") and not f.startswith("_"))

    def log_fail(url, err):
        with open(FAILURES_FILE, "a", encoding="utf-8") as f:
            f.write("%s\t%s\n" % (url, err))

    for i, url in enumerate(urls, 1):
        base_slug = slugify(url)
        path0 = os.path.join(BASE, base_slug + ".txt")
        if os.path.exists(path0):
            with open(path0, encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip()
            if first == "Source: " + url:
                skipped += 1
                continue
        slug = base_slug
        n = 2
        while (slug + "-" + str(n)) in used_slugs or os.path.exists(os.path.join(BASE, slug + "-" + str(n) + ".txt")):
            n += 1
        if slug in used_slugs or os.path.exists(path0):
            slug = slug + "-" + str(n)
        path = os.path.join(BASE, slug + ".txt")

        html, err = fetch(url)
        if html is None:
            failed += 1
            log_fail(url, err or "unknown")
            print("[%d/%d] FAIL %s (%s)" % (i, len(urls), slug, err))
            time.sleep(random.uniform(1.0, 2.0))
            continue

        low = html[:4000].lower()
        if "cf-challenge" in low or "just a moment" in low or "challenge-platform" in low:
            consec_blocks += 1
            failed += 1
            log_fail(url, "blocked (cloudflare challenge)")
            print("[%d/%d] BLOCKED %s" % (i, len(urls), slug))
            if consec_blocks >= 6:
                print("ABORT: repeated Cloudflare blocks; stopping to avoid hammering the site.")
                break
        else:
            consec_blocks = 0

        try:
            title, body = extract(html)
        except Exception as e:
            failed += 1
            log_fail(url, "parse error: %s" % str(e)[:120])
            print("[%d/%d] PARSE-FAIL %s (%s)" % (i, len(urls), slug, e))
            time.sleep(random.uniform(1.0, 2.0))
            continue

        if len(body) < 200:
            failed += 1
            log_fail(url, "content too short (%d chars) - likely non-article page" % len(body))
            print("[%d/%d] SHORT %s (%d chars)" % (i, len(urls), slug, len(body)))
        else:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("Source: %s\nTitle: %s\n\n%s" % (url, title, body))
            os.replace(tmp, path)
            used_slugs.add(slug)
            saved += 1

        if i % 25 == 0 or i == len(urls):
            print("[%d/%d] saved=%d skipped=%d failed=%d" % (i, len(urls), saved, skipped, failed))
        time.sleep(random.uniform(1.0, 2.0))

    print("DONE saved=%d skipped(existing)=%d failed=%d total_urls=%d" % (saved, skipped, failed, len(urls)))


if __name__ == "__main__":
    main()
