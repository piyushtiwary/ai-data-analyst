from __future__ import annotations

from typing import Any, Dict, List

import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

from langchain.tools import tool


@tool
def generate_pdf_report_tool(
    output_path: str,
    title: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a PDF report with structured sections and optional images.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    for index, section in enumerate(sections):
        heading = section.get("heading")
        body = section.get("body")
        images = section.get("images", [])

        if heading:
            story.append(Paragraph(heading, styles["Heading2"]))
            story.append(Spacer(1, 8))

        if body:
            story.append(Paragraph(body, styles["BodyText"]))
            story.append(Spacer(1, 10))

        for image_path in images:
            try:
                normalized = os.path.normpath(image_path)
                story.append(Image(normalized, width=420, height=260))
                story.append(Spacer(1, 12))
            except Exception:
                continue

        if index < len(sections) - 1:
            story.append(PageBreak())

    doc.build(story)

    return {"report_path": output_path}
