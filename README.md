# TabulRAG 📊

**TabulRAG** is a specialized Retrieval-Augmented Generation (RAG) system designed to ingest, parse, and retrieve complex tabular data from scientific PDF documents. 

Unlike standard RAG pipelines that flatten tables into text (destroying structure and semantic meaning), TabulRAG preserves table grids, headers, and scientific symbols (e.g., `< 20 ppm` vs `420 ppm`) to ensure high-accuracy answers for pharmaceutical and technical domains.

## 🌟 Key Features

* **Advanced Table Parsing:** Preserves 2D structure of tables using Markdown format.
* **Multi-Parser Support:** Includes a dedicated benchmarking module to compare:
    * **Marker:** (Deep Learning) **Recommended.** Best for scientific notation, complex layouts, and math symbols.
    * **LlamaParse:** (Cloud AI) High quality, but requires API key.
    * **PyMuPDF:** (Heuristic) Fast, good for simple layouts.
    * **pdfplumber:** (Native) Standard text extraction.
* **Automated Benchmarking:** Generates side-by-side HTML reports to verify parsing accuracy before ingestion.
* **GPU Acceleration:** optimized for Deep Learning parsers (Marker/Surya) on NVIDIA GPUs (e.g., RTX 8000).

## 🛠️ Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone [https://github.com/your-username/TabulRAG.git](https://github.com/your-username/TabulRAG.git)
cd TabulRAG

# Install dependencies
poetry install

# Install deep learning OCR dependencies (for Marker)
poetry run pip install "numpy<2.0"  # Ensures compatibility with Marker/Surya
```

## ⚙️ Configuration
1. Copy the example environment file:
```Bash
cp .env.example .env
```
2. Edit .env to configure your API keys and settings:

```bash
Ini, TOMLOPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=llx-... (Optional, for LlamaParse)
# Choose your parser: marker, llamaparse, pymupdf, or pdfplumber
TABLE_PARSER=marker
```

## 🚀 Pipeline Usage
1. **Benchmark Parsers (Optional but Recommended)**

Before ingesting, run the evaluator to generate an HTML report comparing how different parsers handle your specific PDFs. This helps identify "hallucinations" (e.g., < becoming 4).Bash# Run evaluation on all PDFs in /data
```bash
poetry run python evaluate_tables.py
```
- Output: Check evaluation_reports/ for the HTML report.
- Note: Marker will download models (~2GB) on the first run.
2. **Ingest Documents**
Run the ingestion script to parse PDFs and build the vector index.Bash# Re-ingest documents using settings from .env
poetry run python ingest_pdfs.py
3. **Run the RAG Agent**
Start the interactive query agent.
```Bash
poetry run python rag_agent.py
```
## 📊 Benchmarking Insights
During development, we compared parsers on **Handbook of Pharmaceutical Excipients data**.

| Parser | Symbol Handling | Math Accuracy | Speed | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Marker** | ✅ Excellent | ✅ High | Slower (GPU acc.) | **Champion 🏆** |
| **LlamaParse** | ⚠️ Inconsistent | ⚠️ Med | Fast (Cloud) | Good alternative |
| **pdfplumber** | ❌ Poor | ❌ Low | Very Fast | Baseline only |

> **Critical Finding:** Marker correctly preserved safety limits like `< 20 ppm`, whereas other parsers often hallucinated `420 ppm`.


## 📝 Developer Notes
Fixing Gitignore Encoding (Windows PowerShell) If you encounter encoding issues with .gitignore on Windows:

```PowerShell

(Get-Content .gitignore) | Set-Content -Encoding UTF8 .gitignore

```