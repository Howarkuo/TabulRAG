"""
Table Parsing Evaluation Module

This script systematically evaluates table extraction quality from PDFs using
multiple parsing methods: LlamaParse, PyMuPDF, and pdfplumber.

Features:
- Extract tables using multiple parsing strategies
- Compare against ground truth samples (if available)
- Calculate evaluation metrics
- Generate visual HTML reports
- Export parsed tables for manual inspection
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# PDF parsing libraries
import fitz  # PyMuPDF
import pdfplumber

# Optional: LlamaParse (requires API key)
try:
    from llama_parse import LlamaParse
    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False

# Optional: Marker (requires marker-pdf)
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "./data"
GROUND_TRUTH_DIR = "./ground_truth"
OUTPUT_DIR = "./parsed_tables"
REPORT_DIR = "./evaluation_reports"


class TableParser:
    """Base class for table parsing strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Parse tables from PDF. Returns list of table dictionaries."""
        raise NotImplementedError

    def _extract_markdown_tables(self, text: str) -> List[str]:
        """Extract markdown tables from text"""
        tables = []
        lines = text.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line:
                current_table.append(line)
                in_table = True
            elif in_table:
                if current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                in_table = False
        
        if current_table:
            tables.append('\n'.join(current_table))
        
        return tables
    
    def _markdown_to_dataframe(self, markdown_table: str) -> Optional[pd.DataFrame]:
        """Convert markdown table to DataFrame"""
        try:
            lines = [l.strip() for l in markdown_table.split('\n') if l.strip()]
            
            if len(lines) < 2:
                return None
            
            # Parse header
            header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
            
            # Skip separator line
            data_lines = [l for l in lines[2:] if not all(c in '|-: ' for c in l)]
            
            # Parse data rows
            data = []
            for line in data_lines:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(row) == len(header):
                    data.append(row)
            
            return pd.DataFrame(data, columns=header)
        
        except Exception:
            return None


class PDFPlumberParser(TableParser):
    """Table parser using pdfplumber"""
    
    def __init__(self):
        super().__init__("pdfplumber")
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber"""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for table_num, table_data in enumerate(page_tables, 1):
                    if table_data:
                        print(table_data)
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        tables.append({
                            'parser': self.name,
                            'page': page_num,
                            'table_num': table_num,
                            'data': df,
                            'raw_data': table_data
                        })
        
        return tables


class PyMuPDFParser(TableParser):
    """Table parser using PyMuPDF (fitz)"""
    
    def __init__(self):
        super().__init__("PyMuPDF")
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using PyMuPDF"""
        tables = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Find tables using text extraction
            page_tables = page.find_tables()
            
            for table_num, table in enumerate(page_tables, 1):
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 1:
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        tables.append({
                            'parser': self.name,
                            'page': page_num + 1,
                            'table_num': table_num,
                            'data': df,
                            'raw_data': table_data
                        })
                except Exception as e:
                    print(f"  ⚠️  Error extracting table on page {page_num + 1}: {e}")
        
        doc.close()
        return tables


class LlamaParseParser(TableParser):
    """Table parser using LlamaParse (cloud-based)"""
    
    def __init__(self):
        super().__init__("LlamaParse")
        
        if not LLAMA_PARSE_AVAILABLE:
            raise ImportError("llama-parse not installed")
        
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not set in environment")
        
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction="Extract tables with high precision. IMPORTANT: distinct strictly between the number '4' and the less-than symbol '<'. Do not convert '<' to '4'. Preserve all mathematical symbols and original structure exactly."
        )
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using LlamaParse"""
        tables = []
        
        try:
            # Parse document
            documents = self.parser.load_data(pdf_path)
            
            # Extract tables from markdown
            for doc_num, doc in enumerate(documents, 1):
                # Simple markdown table detection
                text = doc.text
                table_blocks = self._extract_markdown_tables(text)
                
                for table_num, table_md in enumerate(table_blocks, 1):
                    df = self._markdown_to_dataframe(table_md)
                    
                    if df is not None and not df.empty:
                        tables.append({
                            'parser': self.name,
                            'page': doc_num,
                            'table_num': table_num,
                            'data': df,
                            'raw_data': table_md
                        })
        
        except Exception as e:
            print(f"LlamaParse error: {e}")
        
        return tables

class MarkerParser(TableParser):
    """Table parser using Marker (local deep learning model)"""
    
    def __init__(self):
        super().__init__("Marker")
        
        if not MARKER_AVAILABLE:
            raise ImportError("marker-pdf not installed")
            
        print("Loading Marker models")
        self.model_lst = load_all_models()
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using Marker"""
        tables = []
        
        try:
            # Convert PDF to markdown
            full_text, _, _ = convert_single_pdf(pdf_path, self.model_lst)
            
            # Marker outputs the hole doc as markdown, we just need to extract tables
            table_blocks = self._extract_markdown_tables(full_text)
            
            for table_num, table_md in enumerate(table_blocks, 1):
                df = self._markdown_to_dataframe(table_md)
                
                if df is not None and not df.empty:
                    tables.append({
                        'parser': self.name,
                        'page': 1, # Marker merges pages, hard to track unless tracking metadata deeply
                        'table_num': table_num,
                        'data': df,
                        'raw_data': table_md
                    })
                    
        except Exception as e:
            print(f"Marker error: {e}")
            
        return tables
class TableEvaluator:
    """Evaluate table parsing quality"""
    
    def __init__(self):
        self.parsers = self._initialize_parsers()
        self.results = []
    
    def _initialize_parsers(self) -> List[TableParser]:
        """Initialize available parsers"""
        parsers = [
            PDFPlumberParser(),
            PyMuPDFParser()
        ]
        
        # Add LlamaParse if available
        if LLAMA_PARSE_AVAILABLE and os.getenv("LLAMA_CLOUD_API_KEY"):
            try:
                parsers.append(LlamaParseParser())
            except Exception as e:
                print(f"⚠️  Could not initialize LlamaParse: {e}")
        
        # Add Marker if available
        if MARKER_AVAILABLE:
            try:
                parsers.append(MarkerParser())
            except Exception as e:
                print(f"Could not initialize Marker: {e}")
        
        return parsers
    
    def evaluate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Evaluate table parsing for a single PDF"""
        print(f"Processing: {Path(pdf_path).name}")
        
        results = {
            'pdf_name': Path(pdf_path).name,
            'pdf_path': pdf_path,
            'parsers': {}
        }
        
        for parser in self.parsers:
            print(f"Parsing with {parser.name}...")
            
            try:
                tables = parser.parse_pdf(pdf_path)
                
                results['parsers'][parser.name] = {
                    'table_count': len(tables),
                    'tables': tables,
                    'success': True
                }
                
                print(f"Found {len(tables)} table(s)")
                
            except Exception as e:
                results['parsers'][parser.name] = {
                    'table_count': 0,
                    'tables': [],
                    'success': False,
                    'error': str(e)
                }
                print(f"Error: {e}")
        
        return results
    
    def evaluate_all(self) -> List[Dict[str, Any]]:
        """Evaluate all PDFs in data directory"""
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {DATA_DIR}")
            return []
        
        print(f"Found {len(pdf_files)} PDF file(s)")
        
        all_results = []
        
        for pdf_path in pdf_files:
            result = self.evaluate_pdf(str(pdf_path))
            all_results.append(result)
        
        return all_results
    
    def export_tables(self, results: List[Dict[str, Any]]):
        """Export parsed tables to CSV files"""
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting tables to {OUTPUT_DIR}/...")
        
        for result in results:
            pdf_name = Path(result['pdf_name']).stem
            
            for parser_name, parser_result in result['parsers'].items():
                if not parser_result['success']:
                    continue
                
                for table in parser_result['tables']:
                    page = table['page']
                    table_num = table['table_num']
                    df = table['data']
                    
                    # Create filename
                    filename = f"{pdf_name}_p{page}_t{table_num}_{parser_name}.csv"
                    filepath = output_path / filename
                    
                    # Save to CSV
                    df.to_csv(filepath, index=False)
                    print(f"{filename}")
    
    def generate_html_report(self, results: List[Dict[str, Any]]):
        """Generate HTML evaluation report"""
        report_path = Path(REPORT_DIR)
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"table_evaluation_{timestamp}.html"
        
        print(f"\n📊 Generating HTML report: {report_file}")
        
        html = self._build_html_report(results)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Report saved!")
        
        return report_file
    
    def _build_html_report(self, results: List[Dict[str, Any]]) -> str:
        """Build HTML report content"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Table Parsing Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .pdf-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .parser-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .parser-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #2196F3;
        }}
        .parser-card.success {{
            border-left-color: #4CAF50;
        }}
        .parser-card.error {{
            border-left-color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .metric {{
            display: inline-block;
            padding: 5px 15px;
            margin: 5px;
            background: #e3f2fd;
            border-radius: 20px;
            font-weight: bold;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>📊 Table Parsing Evaluation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><span class="metric">PDFs Analyzed: {len(results)}</span></p>
        <p><span class="metric">Parsers Used: {len(self.parsers)}</span></p>
    </div>
"""
        
        # Add results for each PDF
        for result in results:
            html += f"""
    <div class="pdf-section">
        <h2>📄 {result['pdf_name']}</h2>
        
        <div class="parser-comparison">
"""
            
            for parser_name, parser_result in result['parsers'].items():
                status_class = 'success' if parser_result['success'] else 'error'
                status_icon = '✅' if parser_result['success'] else '❌'
                
                html += f"""
            <div class="parser-card {status_class}">
                <h3>{status_icon} {parser_name}</h3>
                <p><strong>Tables Found:</strong> {parser_result['table_count']}</p>
"""
                
                if not parser_result['success']:
                    html += f"<p><strong>Error:</strong> {parser_result.get('error', 'Unknown')}</p>"
                
                html += "</div>\n"
            
            html += "</div>\n"
            
            # Show table previews
            for parser_name, parser_result in result['parsers'].items():
                if parser_result['success'] and parser_result['tables']:
                    html += f"<h3>Tables from {parser_name}</h3>\n"
                    
                    for table in parser_result['tables'][:3]:  # Show first 3 tables
                        df = table['data']
                        page = table['page']
                        table_num = table['table_num']
                        
                        html += f"<h4>Page {page}, Table {table_num}</h4>\n"
                        html += df.head(10).to_html(classes='table', index=False)
            
            html += "</div>\n"
        
        html += """
</body>
</html>
"""
        
        return html


def main():
    """Main execution function"""
    print("=" * 60)
    print("Table Parsing Evaluation")
    print("=" * 60)
    
    # Create output directories
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(REPORT_DIR).mkdir(exist_ok=True)
    Path(GROUND_TRUTH_DIR).mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = TableEvaluator()
    
    print(f"\n🔧 Initialized {len(evaluator.parsers)} parser(s):")
    for parser in evaluator.parsers:
        print(f"  - {parser.name}")
    
    # Evaluate all PDFs
    results = evaluator.evaluate_all()
    
    if not results:
        print("\n⚠️  No results to process")
        return
    
    # Export tables
    evaluator.export_tables(results)
    
    # Generate report
    report_file = evaluator.generate_html_report(results)
    
    print("\n" + "=" * 60)
    print("✨ Evaluation Complete!")
    print("=" * 60)
    print(f"\n📁 Outputs:")
    print(f"  - Parsed tables: {OUTPUT_DIR}/")
    print(f"  - HTML report: {report_file}")
    print(f"\n💡 Next steps:")
    print(f"  1. Open the HTML report in your browser")
    print(f"  2. Review parsed tables in {OUTPUT_DIR}/")
    print(f"  3. Create ground truth samples in {GROUND_TRUTH_DIR}/")


if __name__ == "__main__":
    main()
