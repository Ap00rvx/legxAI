import json
import time
import requests
from urllib.parse import urljoin, urlencode, urlsplit, urlunsplit, parse_qsl
from bs4 import BeautifulSoup

BASE = "https://indiankanoon.org"
# Start directly at the Supreme Court browse page you shared
START_URL = f"{BASE}/browse/supremecourt/"

# Tweakable settings
MIN_YEAR = 1950
MAX_YEAR = 2025  # include recent years
MAX_PAGES_PER_YEAR = 200  # safety cap; will stop earlier when no results
REQUEST_DELAY_SEC = 0.5   # be polite
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
}


def add_or_replace_query(url: str, params: dict) -> str:
    """Return URL with params merged into its query string."""
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in params.items()})
    new_query = urlencode(q)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def get_soup(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"Warning: HTTP {resp.status_code} for {url}")
            return None
        return BeautifulSoup(resp.content, "html.parser")
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None


def iter_year_links(soup: BeautifulSoup):
    """Yield (year:int, href:str) from the Supreme Court browse page."""
    for block in soup.find_all(class_="browselist"):
        a = block.find("a")
        if not a:
            continue
        text = (a.text or "").strip()
        if text.isdigit():
            year = int(text)
            href = a.get("href")
            if href and MIN_YEAR <= year <= MAX_YEAR:
                yield year, href


def collect_supreme_court_links():
    session = requests.Session()
    all_links: list[str] = []

    print(f"Loading years from: {START_URL}")
    soup = get_soup(session, START_URL)
    if not soup:
        print("Failed to load the Supreme Court browse page.")
        return all_links

    year_entries = sorted(set(iter_year_links(soup)), key=lambda x: x[0])
    if not year_entries:
        print("No year links were found on the page. Structure may have changed.")
        return all_links

    for year, href in year_entries:
        year_url = urljoin(BASE, href)
        print(f"{year} Year Started ..... {year_url}")

        # Paginate directly on the year page until no results
        for page_num in range(MAX_PAGES_PER_YEAR):
            time.sleep(REQUEST_DELAY_SEC)

            page_url = year_url
            # add pagenum for pages after the first
            if page_num > 0:
                page_url = add_or_replace_query(year_url, {"pagenum": page_num})

            page_soup = get_soup(session, page_url)
            if not page_soup:
                print(f"Stopping year {year}: failed to load page {page_num}.")
                break

            result_links = page_soup.find_all(class_="result_url")
            if not result_links:
                # No more results for this year
                print(f"{year} Year Completed (last page: {page_num - 1})")
                break

            batch = 0
            for r in result_links:
                href = r.get("href")
                if not href:
                    continue
                case_url = urljoin(BASE, href)
                all_links.append(case_url)
                batch += 1

            print(f"  {year} p{page_num}: collected {batch} case links")

    return all_links


def main():
    links = {"Supreme Court of India": collect_supreme_court_links()}

    if not links["Supreme Court of India"]:
        print("Warning: Supreme Court of India produced no links.")

    out_path = "links_Supreme_Court.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=4, ensure_ascii=False)
    print(f"Saved {out_path} with {len(links['Supreme Court of India'])} links")


if __name__ == "__main__":
    main()