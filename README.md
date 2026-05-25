# AI Data Analyst

Automated, multi-agent data analysis and reporting pipeline for tabular datasets. It focuses on evidence-driven EDA, model evaluation, and HTML reporting without exposing raw data to the LLM.

## What It Does

- Runs a structured analysis flow (EDA, statistical testing, modeling, validation, visualization, reporting).
- Produces an HTML report and PDF (via Playwright) with charts and narrative explanations.
- Tracks results using tool outputs and metadata only (no raw data access for the LLM).

## Key Features (How They Work)

- Multi-agent graph orchestration: A LangGraph pipeline sequences EDA reasoning, ML evaluation, visualization, and report generation, passing only metadata and tool outputs between nodes.
- Leakage and validation checks: The ML stage runs checks for leakage risks and stability; findings are surfaced in a dedicated report section.
- Class-imbalance handling: SMOTE and threshold optimization improve decisioning on imbalanced targets.
- Statistical testing + hypotheses: t-test, Mann-Whitney, ANOVA, chi-square, and correlation significance feed into the analytical narrative.
- Visualization + narrative pairing: Each chart is generated from processed data and explained with evidence-based commentary.
- HTML + Playwright rendering: The report is built from a Jinja2 template and rendered to PDF using Playwright.

## What Makes It Special

- Evidence-first reporting: conclusions are tied to quantitative tests and model metrics.
- LLM safety boundary: the LLM never sees raw data or model internals.
- Decision-ready output: includes key findings cards, limitations, and threshold guidance.

## Walkthrough

1) Create environment config
 - Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
 - Set `LLM_PROVIDER` and `LLM_MODEL` if you want a specific model.
2) Install dependencies
 - `pip install -r requirements.txt`
3) Install Playwright (for PDF output)
 - `playwright install`
4) Set the dataset
 - Place your CSV in `data/` and update the file path in [main.py](main.py).
5) Run the pipeline
 - `python main.py`
6) Review outputs
 - HTML report: `outputs/report.html`
 - PDF report: generated via Playwright (if enabled)
