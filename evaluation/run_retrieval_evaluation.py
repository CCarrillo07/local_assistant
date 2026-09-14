import json
from pathlib import Path

from config import CONFIG
from rag.embedder import OllamaEmbedder
from rag.service import RAGService
from rag.vector_store_factory import create_vector_store

CASES_PATH = Path(__file__).with_name(
    "rag_cases.json"
)

def load_cases() -> list[dict]:
    """Load the retrieval questions and expected results."""
    
    return json.loads(
        CASES_PATH.read_text(
            encoding="utf-8"
        )
    )
    
def main() -> int:
    """Run the retrieval evaluation without calling the LLM."""
    
    rag_config = CONFIG.rag
    
    embedder = OllamaEmbedder(
        model=rag_config.embedding_model,
        query_prefix=rag_config.embedding_query_prefix,
        document_prefix=rag_config.embedding_document_prefix
    )
    
    vector_store = create_vector_store(
        config=rag_config,
        embedder=embedder
    )
    
    rag_service = RAGService(
        document_directory=rag_config.document_directory,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=rag_config.chunk_size,
        overlap=rag_config.overlap,
        top_k=rag_config.top_k,
        min_score=rag_config.min_score
    )
    
    cases = load_cases()
    passed_cases = 0
    
    try:
        rag_service.initialize()
        
        for case in cases:
            question = case["question"]
            expected_source = case["expected_source"]
            expected_phrase = case["expected_phrase"]
            
            forbidden_sources = set(
                case["forbidden_sources"]
            )
            
            results = vector_store.search(
                query=question,
                top_k=rag_config.top_k,
                min_score=rag_config.min_score
            )
            
            # A null expected source means that the question
            # should not retrieve any document chunks
            if expected_source is None:
                case_passed = not results
                
            else:
                top_result = (
                    results[0]
                    if results
                    else None
                )
                
                correct_source = (
                    top_result is not None
                    and top_result.chunk.source
                    == expected_source
                )
                
                correct_phrase = (
                    top_result is not None
                    and expected_phrase.lower()
                    in top_result.chunk.text.lower()
                )
                
                forbidden_found = any(
                    result.chunk.source
                    in forbidden_sources
                    for result in results
                )
                
                case_passed = (
                    correct_source
                    and correct_phrase
                    and not forbidden_found
                )
                
            status = (
                "PASS" 
                if case_passed
                else "FAIL"
            )
            
            print(f"\n[{status}]{question}")
            
            if not results:
                print(" No chunks retrieveed")
                
            for position, result in enumerate(
                results,
                start=1
            ):
                print(
                    f" {position}. "
                    f"{result.chunk.source}, "
                    f"chunk {result.chunk.chunk_index}, "
                    f"score={result.score:.4f}"
                )
                
            if case_passed:
                passed_cases +=1
    
    finally:
        # Qdrant needs to release its local storage lock.
        close_method = getattr(
            vector_store,
            "close",
            None
        )
        
        if callable(close_method):
            close_method()
            
    print(
        f"\nRertrieval evaluation: "
        f"{passed_cases}/{len(cases)} passed"
    )
    
    return (
        0
        if passed_cases == len(cases)
        else 1
    )
    
if __name__ == "__main__":
    raise SystemExit(main())