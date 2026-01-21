# TabulRAG

Table Parsing Benchmark: A dedicated module (evaluate_tables.py) to compare extraction quality across:

- LlamaParse (State-of-the-art AI parsing)
- Marker (Deep learning based)
- PyMuPDF (Fast, heuristic-based)
- pdfplumber (Python native)

##  Pipeline
1. source .env.example
2. python ingest_pdfs.py #re-ingest documents with the new settings
3. python evaluate_tables.py #to generate an HTML report comparing how different parsers handle your specific PDFs.

### Self Note: To write gitignore
 (Get-Content .gitignore) | Set-Content -Encoding UTF8 .gitignore