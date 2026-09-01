"""Capture real screenshots of the deployed dashboard into ../figures/.

The four tabs of the live application become Figures in Chapter 7. Run:

    python screenshots.py                     # uses the deployed URL
    python screenshots.py http://localhost:5173

Requires Playwright's Chromium (the same one render.py uses).
"""
from __future__ import annotations

import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "https://10pearlsaqi.me"

# tab label fragment -> output file stem
TABS = [
    ("Forecast", "ui-forecast"),
    ("Model Evaluation", "ui-eval"),
    ("Monitoring", "ui-monitoring"),
    ("What-If", "ui-whatif"),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(5000)  # let the intro animation finish and data load

        for label, stem in TABS:
            button = page.locator("nav.tabs button", has_text=label)
            if button.count() == 0:
                print(f"  ! tab '{label}' not present (static mode?) - skipped")
                continue
            button.first.click()
            page.wait_for_timeout(4500)  # charts, map tiles and any fetch

            if stem == "ui-whatif":
                # Move the sliders and run the simulation, so the figure shows a
                # real baseline-versus-scenario result rather than an untouched
                # form. Clicking the track sets the value and fires the change
                # event that a controlled React input listens for.
                sliders = page.locator("input[type=range]")
                # a heavy-pollution, still-air scenario: PM2.5 and AQI up, wind down
                for index, fraction in ((0, 0.70), (1, 0.62), (2, 0.05)):
                    if sliders.count() > index:
                        box = sliders.nth(index).bounding_box()
                        page.mouse.click(box["x"] + box["width"] * fraction,
                                         box["y"] + box["height"] / 2)
                        page.wait_for_timeout(300)
                run = page.locator("button", has_text="Run simulation")
                if run.count():
                    run.first.click()
                    page.wait_for_timeout(9000)

            # Grow the viewport to the page height rather than using full_page:
            # the responsive charts re-measure on a resize, but are not redrawn
            # for a stitched full-page capture, which leaves them clipped.
            height = page.evaluate("document.body.scrollHeight")
            page.set_viewport_size({"width": 1440, "height": min(int(height) + 40, 3000)})
            page.wait_for_timeout(2500)
            out = OUT / f"{stem}.png"
            page.screenshot(path=str(out))
            page.set_viewport_size({"width": 1440, "height": 900})
            page.wait_for_timeout(800)
            print(f"  captured {label} -> figures/{out.name}")

        browser.close()
    print("done.")


if __name__ == "__main__":
    main()
