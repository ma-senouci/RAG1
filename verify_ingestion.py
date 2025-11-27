from rag_logic import RAGManager

def main():
    manager = RAGManager(chunk_size=750, chunk_overlap=75)
    print("Starting ingestion...")
    chunks = manager.sync_index("me")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Total vectors in FAISS index: {manager.index.ntotal}")
    
    if chunks:
        print("\nFirst chunk preview (100 chars):")
        print("-" * 30)
        print(f"{chunks[0][:100]}...")
        print("-" * 30)
    else:
        print("No chunks found. Check source files.")

if __name__ == "__main__":
    main()
