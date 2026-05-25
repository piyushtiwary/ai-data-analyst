from __future__ import annotations

from typing import Any, Dict

from pathlib import Path

import os

from jinja2 import Template
from playwright.sync_api import sync_playwright

from langchain.tools import tool

DEFAULT_CSS = """
@page {
  size: A4;
  margin: 22mm 18mm;
}
body {
  font-family: "Segoe UI", "Calibri", "Arial", sans-serif;
  color: #1a1a1a;
  line-height: 1.45;
  font-size: 11pt;
}
 .key-findings {
  margin: 6mm 0 8mm 0;
 }
 .cards {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4mm;
 }
 .card {
  border: 1px solid #e6e6e6;
  background: #f9fafb;
  padding: 4mm;
  border-radius: 6px;
  font-size: 10.5pt;
 }
 .card-title {
  font-weight: 700;
  font-size: 10pt;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #333;
  margin-bottom: 2mm;
 }
main {
  max-width: 750px;
  margin: 0 auto;
}
h1 {
  font-size: 22pt;
  margin: 0 0 6mm 0;
}
h2 {
  font-size: 15pt;
  margin: 6mm 0 3mm 0;
  padding-bottom: 2mm;
  border-bottom: 1px solid #e0e0e0;
}
p {
  margin: 0 0 3mm 0;
}
ul {
  margin: 0 0 4mm 5mm;
}
figure {
  margin: 4mm 0 6mm 0;
}
figcaption {
  font-size: 9.5pt;
  color: #4a4a4a;
  margin-top: 2mm;
}
img {
  width: 100%;
  height: auto;
  border: 1px solid #e6e6e6;
  padding: 2mm;
  background: #fafafa;
}
.section-note {
  color: #555;
  font-size: 10pt;
}
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>{{ css }}</style>
</head>
<body>
  {{ html_content | safe }}
</body>
</html>
"""


@tool
def generate_pdf_from_html_tool(
    output_pdf_path: str,
    output_html_path: str,
    title: str,
    html_content: str,
    css: str | None = None,
) -> Dict[str, Any]:
    """
    Render an HTML report to PDF using Playwright with optional CSS styling.
    Requires Playwright browsers (run `playwright install`).
    """
    if not html_content:
        html_content = "<main><h1>Report</h1><p>No content provided.</p></main>"

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    template = Template(HTML_TEMPLATE)
    html_doc = template.render(
        title=title,
        css=css or DEFAULT_CSS,
        html_content=html_content,
    )

    with open(output_html_path, "w", encoding="utf-8") as handle:
        handle.write(html_doc)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        file_url = Path(output_html_path).resolve().as_uri()
        page.goto(file_url, wait_until="load")
        page.pdf(path=output_pdf_path, format="A4", print_background=True)
        browser.close()

    return {
        "report_path": output_pdf_path,
        "html_report_path": output_html_path,
    }
