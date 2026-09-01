"""Render every *.html diagram in this folder to a high-resolution PNG in
../figures/.  Uses the Chromium that ships with Playwright.

Run:  python render.py
"""
from __future__ import annotations
import pathlib
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
SCALE = 3  # device pixel ratio -> crisp, print-quality output


def main() -> None:
    htmls = sorted(HERE.glob("*.html"))
    if not htmls:
        print("no .html diagrams found")
        return
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=SCALE)
        for html in htmls:
            page.goto(html.as_uri())
            page.wait_for_timeout(150)
            target = page.query_selector("#canvas") or page
            out = OUT / (html.stem + ".png")
            target.screenshot(path=str(out))
            print(f"  rendered {html.name} -> figures/{out.name}")
        browser.close()
    print("done.")


if __name__ == "__main__":
    main()
