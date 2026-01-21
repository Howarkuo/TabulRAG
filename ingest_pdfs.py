"""
PDF Ingestion Script for RAG Agent with Table Parsing

This script loads PDF documents from the data/ directory, extracts tables
with preserved structure, and builds a vector index using OpenAI embeddings.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Table parsing libraries
import fitz  # PyMuPDF
import pdfplumber

# Optional: LlamaParse (requires API key)
try:
    from llama_parse import LlamaParse
    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "./data"
STORAGE_DIR = "./storage_non_tables"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TABLE_PARSER = os.getenv("TABLE_PARSER", "pdfplumber")  # Options: pdfplumber, pymupdf, llamaparse
EXTRACT_TABLES = os.getenv("EXTRACT_TABLES", "true").lower() == "true"


def setup_llama_index():
    """Configure LlamaIndex global settings"""
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def extract_tables_pdfplumber(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using pdfplumber"""
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            
            for table_num, table_data in enumerate(page_tables, 1):
                if table_data and len(table_data) > 1:
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        # Convert to markdown
                        markdown = df.to_markdown(index=False)
                        
                        tables.append({
                            'page': page_num,
                            'table_num': table_num,
                            'markdown': markdown,
                            'rows': len(df),
                            'cols': len(df.columns)
                        })
                    except Exception as e:
                        print(f"  ⚠️  Error parsing table on page {page_num}: {e}")
    
    return tables


def extract_tables_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using PyMuPDF"""
    tables = []
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_tables = page.find_tables()
        
        for table_num, table in enumerate(page_tables, 1):
            try:
                table_data = table.extract()
                
                if table_data and len(table_data) > 1:
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    
                    # Convert to markdown
                    markdown = df.to_markdown(index=False)
                    
                    tables.append({
                        'page': page_num + 1,
                        'table_num': table_num,
                        'markdown': markdown,
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
            except Exception as e:
                print(f"  ⚠️  Error parsing table on page {page_num + 1}: {e}")
    
    doc.close()
    return tables


def extract_tables_llamaparse(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using LlamaParse (cloud-based, best quality)"""
    tables = []
    
    if not LLAMA_PARSE_AVAILABLE:
        print("  ⚠️  llama-parse not installed, falling back to pdfplumber")
        return extract_tables_pdfplumber(pdf_path)
    
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("  ⚠️  LLAMA_CLOUD_API_KEY not set, falling back to pdfplumber")
        return extract_tables_pdfplumber(pdf_path)
    
    try:
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction="Extract tables with high precision. IMPORTANT: distinct strictly between the number '4' and the less-than symbol '<'. Do not convert '<' to '4'. Preserve all mathematical symbols and original structure exactly."
        )
        
        # Parse document
        documents = parser.load_data(pdf_path)
        
        # Extract tables from markdown
        for doc_num, doc in enumerate(documents, 1):
            text = doc.text
            table_blocks = _extract_markdown_tables(text)
            
            for table_num, table_md in enumerate(table_blocks, 1):
                df = _markdown_to_dataframe(table_md)
                
                if df is not None and not df.empty:
                    tables.append({
                        'page': doc_num,
                        'table_num': table_num,
                        'markdown': table_md,
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
    
    except Exception as e:
        print(f"  ⚠️  LlamaParse error: {e}, falling back to pdfplumber")
        return extract_tables_pdfplumber(pdf_path)
    
    return tables


def _extract_markdown_tables(text: str) -> List[str]:
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


def _markdown_to_dataframe(markdown_table: str) -> pd.DataFrame:
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


def extract_text_with_tables(pdf_path: str, parser: str = "pdfplumber") -> str:
    """Extract text and tables from PDF, combining them intelligently"""
    
    # Extract tables based on chosen parser
    if parser == "pymupdf":
        tables = extract_tables_pymupdf(pdf_path)
    elif parser == "llamaparse":
        tables = extract_tables_llamaparse(pdf_path)
    else:  # default to pdfplumber
        tables = extract_tables_pdfplumber(pdf_path)
    
    # Extract regular text using pdfplumber
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Get page text
            page_text = page.extract_text() or ""
            
            # Find tables for this page
            page_tables = [t for t in tables if t['page'] == page_num]
            
            # If there are tables on this page, add them in markdown format
            if page_tables:
                full_text.append(f"\n--- Page {page_num} ---\n")
                full_text.append(page_text)
                full_text.append(f"\n\n📊 Tables on Page {page_num}:\n")
                
                for table in page_tables:
                    full_text.append(f"\nTable {table['table_num']} ({table['rows']} rows × {table['cols']} columns):\n")
                    full_text.append(table['markdown'])
                    full_text.append("\n")
            else:
                full_text.append(f"\n--- Page {page_num} ---\n")
                full_text.append(page_text)
    
    return "\n".join(full_text)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages)

def ingest_pdfs_with_tables():
    """Load PDFs with table extraction and create vector index"""
    print("🔍 Loading PDF documents with table extraction...")
    print(f"📊 Using table parser: {TABLE_PARSER}")
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found!")
    
    # Get all PDF files
    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")
    
    print(f"✅ Found {len(pdf_files)} PDF file(s)")
    
    # Process each PDF
    documents = []
    total_tables = 0
    
    for pdf_path in pdf_files:
        print(f"\n  📄 Processing: {pdf_path.name}")
        
        # Determine extraction strategy
        if EXTRACT_TABLES:
            # Extract text with tables
            if TABLE_PARSER == "pymupdf":
                tables = extract_tables_pymupdf(str(pdf_path))
            elif TABLE_PARSER == "llamaparse":
                tables = extract_tables_llamaparse(str(pdf_path))
            else:
                tables = extract_tables_pdfplumber(str(pdf_path))
            
            print(f"Found {len(tables)} table(s)")
            total_tables += len(tables)
            
            # Extract combined text
            text_content = extract_text_with_tables(str(pdf_path), TABLE_PARSER)
        else:
            # Extract plain text only
            print("Table extraction disabled, extracting plain text only.")
            tables = []
            text_content = extract_text_from_pdf(str(pdf_path))
        
        # Create LlamaIndex document
        doc = Document(
            text=text_content,
            metadata={
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "table_count": len(tables),
                "parser": TABLE_PARSER if EXTRACT_TABLES else "none",
                "structured_tables": EXTRACT_TABLES
            }
        )
        
        documents.append(doc)
    
    print(f"\n✅ Loaded {len(documents)} document(s) with {total_tables} total table(s)")
    
    print("\n🔨 Building vector index...")
    
    # Create vector index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    print(f"\n💾 Persisting index to '{STORAGE_DIR}'...")
    
    # Persist index to disk
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    print("Index created and saved successfully!")
    print(f"Index Statistics:")
    print(f"  - Total documents: {len(documents)}")
    print(f"  - Table extraction enabled: {EXTRACT_TABLES}")
    if EXTRACT_TABLES:
        print(f"  - Total tables extracted: {total_tables}")
        print(f"  - Table parser: {TABLE_PARSER}")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP}")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    
    return index


def main():
    """Main execution function"""
    print("=" * 60)
    print("RAG Agent - PDF Ingestion with Table Parsing")
    print("=" * 60)
    print()
    
    # Setup
    setup_llama_index()
    
    # Ingest PDFs
    try:
        index = ingest_pdfs_with_tables()
        print("\n✨ Ingestion complete! You can now run rag_agent.py to query.")
        print("\n💡 Tables are preserved in markdown format for better retrieval.")
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
