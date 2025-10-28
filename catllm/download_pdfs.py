#!/usr/bin/env python3
"""
download_pdfs.py — v3

- Detects PDFs by Content-Type / Content-Disposition (handles viewcontent.cgi?article=...).
- Include external PDFs via --include-external.
- Parallel downloads (--concurrency N).
- Resumable when server supports Range (--resume).
- Organizes external PDFs by domain.

Example:
  python download_pdfs.py \
    "https://digitalcommons.unl.edu/animalscinbcr/" ./pdfs \
    --include-external --concurrency 6 --resume --delay 1.0 --max-pages 1000
"""

import os, re, time, argparse
from urllib.parse import urljoin, urlparse, urldefrag
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib.robotparser

USER_AGENT = "pdf-crawler-bot/3.0 (+https://example.com)"
PDF_URL_HINT = re.compile(r'\.pdf($|\?|#)', re.IGNORECASE)
CD_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)')

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-()" else "_" for c in name).strip()

def is_same_domain(start_netloc: str, url: str) -> bool:
    try:
        return urlparse(url).netloc == start_netloc
    except Exception:
        return False

def allowed_by_robots(robots, url: str) -> bool:
    try:
        return robots.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def find_links(html_text: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "mailto:")):
            continue
        full = urldefrag(urljoin(base_url, href)).url
        links.add(full)
    return links

def head_like(session: requests.Session, url: str, timeout=20) -> requests.Response | None:
    """Try HEAD first, some servers dislike it; fall back to GET (no stream) if needed."""
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code < 400:
            return r
    except Exception:
        pass
    try:
        r = session.get(url, allow_redirects=True, timeout=timeout, stream=False)
        if r.status_code < 400:
            return r
    except Exception:
        pass
    return None

def looks_like_pdf_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/pdf" in ct:
        return True
    # Some sites send generic octet-stream but with a PDF filename
    cd = resp.headers.get("Content-Disposition") or ""
    if "attachment" in cd.lower():
        m = CD_FILENAME.search(cd)
        if m and m.group(1).lower().endswith(".pdf"):
            return True
    # As a final hint, accept URLs that clearly end with .pdf even if CT is odd
    if PDF_URL_HINT.search(resp.url):
        return True
    return False

def pick_filename(url: str, resp: requests.Response | None) -> str:
    # Prefer Content-Disposition filename if reasonable
    if resp is not None:
        cd = resp.headers.get("Content-Disposition") or ""
        m = CD_FILENAME.search(cd)
        if m:
            name = sanitize_filename(m.group(1))
            if not name.lower().endswith(".pdf"):
                name += ".pdf"
            return name
    # Otherwise from URL path
    name = os.path.basename(urlparse(url).path) or "download.pdf"
    name = sanitize_filename(name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name

def choose_output_path(url: str, out_dir: str, resp: requests.Response | None, group_by_domain=True) -> Path:
    parsed = urlparse(url)
    fname = pick_filename(url, resp)
    base = Path(out_dir) / (parsed.netloc if group_by_domain else "") / fname
    base.parent.mkdir(parents=True, exist_ok=True)
    p = base
    i = 1
    while p.exists():
        p = base.with_name(f"{base.stem}({i}){base.suffix}")
        i += 1
    return p

def resume_download(session: requests.Session, url: str, target_path: Path, delay: float):
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    start_size = tmp_path.stat().st_size if tmp_path.exists() else 0

    # Probe for length and resume support
    info = head_like(session, url)
    total_len = int((info.headers.get("Content-Length") or "0")) if info else 0
    accept_ranges = "bytes" in (info.headers.get("Accept-Ranges") or "").lower() if info else False
    headers = {}
    if accept_ranges and total_len and start_size < total_len and start_size > 0:
        headers["Range"] = f"bytes={start_size}-"

    try:
        with session.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            # If headers suggested a better filename and we're at 0, adjust names before writing
            if start_size == 0:
                cd = r.headers.get("Content-Disposition") or (info.headers.get("Content-Disposition") if info else "")
                m = CD_FILENAME.search(cd or "")
                if m:
                    suggested = sanitize_filename(m.group(1))
                    if not suggested.lower().endswith(".pdf"):
                        suggested += ".pdf"
                    new_target = target_path.with_name(suggested)
                    if not new_target.exists() and not tmp_path.exists():
                        target_path = new_target
                        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
                        target_path.parent.mkdir(parents=True, exist_ok=True)

            remote_len = int(r.headers.get("Content-Length") or "0")
            total = total_len if total_len else (start_size + remote_len if remote_len else None)
            mode = "ab" if "Range" in headers else "wb"
            downloaded = start_size

            with open(tmp_path, mode) as f, tqdm(
                total=total,
                initial=downloaded if total else 0,
                unit="B", unit_scale=True, desc=f"DL {target_path.name}", leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        bar.update(len(chunk))

        if total_len and downloaded < total_len:
            time.sleep(delay)
            return False, target_path

        tmp_path.replace(target_path)
        time.sleep(delay)
        return True, target_path

    except Exception as e:
        print(f"[error] {url} -> {target_path}: {e}")
        time.sleep(delay)
        return False, target_path

def download_worker(args):
    (url, out_dir, session, delay, group_by_domain) = args
    # Use HEAD/GET to obtain naming + confirm it's a PDF
    probe = head_like(session, url)
    if not probe or not looks_like_pdf_response(probe):
        return False, url, "not a PDF (skipped)"
    target = choose_output_path(url, out_dir, probe, group_by_domain=group_by_domain)
    ok, final_path = resume_download(session, url, target, delay)
    return ok, url, str(final_path)

def crawl_and_download(seed_url: str, out_dir: str, max_pages=1000, delay=1.0,
                       include_external=False, concurrency=4, resume=True):
    parsed = urlparse(seed_url)
    start_domain = parsed.netloc
    base_origin = f"{parsed.scheme}://{start_domain}"

    # robots.txt (for HTML pages)
    robots_url = urljoin(base_origin, "/robots.txt")
    robots = urllib.robotparser.RobotFileParser()
    try:
        robots.set_url(robots_url)
        robots.read()
    except Exception:
        print(f"[warn] couldn't read robots.txt at {robots_url} — proceeding cautiously")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    # A Referer header sometimes helps CDNs behind viewcontent endpoints
    session.headers.update({"Referer": seed_url})

    to_visit = [seed_url]
    visited = set()
    candidate_links = set()   # all links we’ll probe for PDF-ness
    pages_visited = 0

    while to_visit and pages_visited < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if not allowed_by_robots(robots, url):
            print(f"[robots] skipping disallowed URL: {url}")
            continue

        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"[warn] failed to fetch {url}: {e}")
            continue

        pages_visited += 1
        links = find_links(r.text, url)
        for link in links:
            # We’ll probe everything; but only enqueue more HTML pages on same domain.
            candidate_links.add(link)
            if is_same_domain(start_domain, link):
                scheme2 = urlparse(link).scheme
                if scheme2 in ("http", "https") and link not in visited and link not in to_visit:
                    # Avoid obvious binaries when crawling HTML pages (saves time)
                    if not PDF_URL_HINT.search(link):
                        to_visit.append(link)

        time.sleep(delay)

    print(f"[crawl] pages visited: {pages_visited}, candidates to probe: {len(candidate_links)}")

    # Filter to PDF links using parallel HEAD/GET probes
    pdf_links = set()
    def probe_worker(url):
        # respect external inclusion choice
        if not include_external and not is_same_domain(start_domain, url):
            return None
        # Quick skip of common non-HTTP schemes
        if urlparse(url).scheme not in ("http", "https"):
            return None
        resp = head_like(session, url)
        if resp and looks_like_pdf_response(resp):
            return resp.url  # use final URL after redirects
        return None

    with ThreadPoolExecutor(max_workers=max(2, concurrency)) as ex:
        futs = {ex.submit(probe_worker, u): u for u in candidate_links}
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                pdf_links.add(res)

    print(f"[filter] PDFs identified: {len(pdf_links)}")

    # Download PDFs (parallel)
    group_by_domain = include_external
    os.makedirs(out_dir, exist_ok=True)
    success = fail = 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        jobs = [
            ex.submit(download_worker, (pdf_url, out_dir, session, delay, group_by_domain))
            for pdf_url in sorted(pdf_links)
        ]
        for fut in as_completed(jobs):
            ok, url, path = fut.result()
            if ok:
                success += 1
                print(f"[ok] {url}  ->  {path}")
            else:
                fail += 1
                print(f"[skip/fail] {url}  ->  {path}")

    print(f"[summary] downloaded: {success}, failed/skipped: {fail}, total PDFs: {len(pdf_links)}")

def main():
    ap = argparse.ArgumentParser(description="Crawl a site and download PDFs (content-type aware).")
    ap.add_argument("start_url", help="Start URL (e.g. https://digitalcommons.unl.edu/animalscinbcr/)")
    ap.add_argument("output_dir", help="Directory to save PDFs")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default 1.0)")
    ap.add_argument("--max-pages", type=int, default=1000, help="Max HTML pages to crawl")
    ap.add_argument("--include-external", action="store_true", help="Include PDFs on other domains")
    ap.add_argument("--concurrency", type=int, default=6, help="Parallelism for probes/downloads")
    ap.add_argument("--resume", action="store_true", help="Try to resume partial downloads")
    args = ap.parse_args()

    crawl_and_download(
        args.start_url, args.output_dir,
        max_pages=args.max_pages, delay=args.delay,
        include_external=args.include_external,
        concurrency=args.concurrency, resume=args.resume
    )

if __name__ == "__main__":
    main()