"""
RAG Agent - Query Engine

This script provides a query interface to ask questions about your PDF documents
using GPT-4o-mini and semantic search with LlamaIndex.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import pandas as pd
from tqdm import tqdm
# Load environment variables
load_dotenv()

# Configuration
STORAGE_DIR = "./storage"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


def setup_llama_index():
    """Configure LlamaIndex global settings"""
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)


def load_index():
    """Load the persisted vector index"""
    if not Path(STORAGE_DIR).exists():
        raise FileNotFoundError(
            f"Storage directory '{STORAGE_DIR}' not found! "
            "Please run 'python ingest_pdfs.py' first to create the index."
        )
    
    print("📂 Loading vector index...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ Index loaded successfully!")
    
    return index

def query_rag_simple(query_engine, question: str):
    response = query_engine.query(question)
    return response.response


def query_rag(query_engine, question: str):
    """Query the RAG system and display results"""
    print(f"\n❓ Question: {question}")
    print("\n🤔 Thinking...\n")
    
    response = query_engine.query(question)
    
    print("=" * 60)
    print("💡 Answer:")
    print("=" * 60)
    print(response.response)
    print()
    
    # Display source information
    if hasattr(response, 'source_nodes') and response.source_nodes:
        print("=" * 60)
        print("📚 Sources:")
        print("=" * 60)
        for i, node in enumerate(response.source_nodes, 1):
            filename = node.metadata.get('file_name', 'Unknown')
            page = node.metadata.get('page_label', 'N/A')
            score = node.score if hasattr(node, 'score') else 'N/A'
            
            print(f"\n[{i}] {filename} (Page: {page}, Relevance: {score:.3f})")
            print(f"    {node.text[:200]}...")
    
    print("\n" + "=" * 60)
    
    return response


def interactive_mode(query_engine):
    """Run in interactive mode for multiple queries"""
    print("\n" + "=" * 60)
    print("🤖 RAG Agent - Interactive Mode")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    print()
    
    while True:
        try:
            question = input("\n💬 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            query_rag(query_engine, question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("RAG Agent - Query Interface")
    print("=" * 60)
    print(f"Model: {LLM_MODEL}")
    print(f"Top-K Retrieval: {TOP_K}")
    print(f"Temperature: {TEMPERATURE}")
    print()
    
    # Setup
    setup_llama_index()
    
    # Load index
    try:
        index = load_index()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact"
    )
    
    # Check if query provided as command line argument
    df = pd.read_csv("question_answer.csv")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        question = row['Question']
        answer = query_rag_simple(query_engine, question)
        df.at[index, 'RAG Answer'] = answer

    df.to_csv("question_answer_storage.csv", index=False)
    #if len(sys.argv) > 1:
    #    question = " ".join(sys.argv[1:])
    #    query_rag(query_engine, question)
    #else:
    #    # Run in interactive mode
    #    interactive_mode(query_engine)


if __name__ == "__main__":
    main()
