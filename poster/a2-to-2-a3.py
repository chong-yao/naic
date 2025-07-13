from playwright.sync_api import sync_playwright

def html_to_png(html_file, output_file, width=2480, height=3508, scale=1):
    """
    Converts an HTML file to PNG.
    Default resolution: A4 at 300 DPI (2480x3508 px)
    Increase scale for higher quality.
    """

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={'width': width, 'height': height, 'device_scale_factor': scale}
        )

        # Load the local HTML file
        page.goto(f"file:///{html_file}", wait_until="load")

        # Wait for all elements/fonts to load (optional)
        page.wait_for_timeout(1000)

        # Take a screenshot of the full page
        page.screenshot(path=output_file, full_page=True)

        browser.close()

# Example usage
html_file = "poster.html"
output_file = "poster.png"
html_to_png(html_file, output_file, width=70, height=99, scale=(7016/99))