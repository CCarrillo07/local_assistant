import json
from pathlib import Path

from config import CONFIG
from llm.ollama_provider import OllamaProvider
from main import SYSTEM_PROMPT
from rag.embedder import OllamaEmbedder
from rag.reranker_factory import create_reranker
from rag.vector_store_factory import create_vector_store

from assistant import Assistant
from evaluation.answer_grounding import (
    ABSTENTION_MESSAGE,
    AnswerEvaluation,
    evaluate_answer
)
from rag.service import RAGService

CASES_PATH  = Path(__file__).with_name(
    "answer_cases.json"
)

def load_cases() -> list[dict]:
    """Load the answer-grounding evaluation cases."""

    return json.loads(
        CASES_PATH.read_text(
            encoding="utf-8"
        )
    )

def generate_answer(
    question: str,
    rag_service: RAGService,
    assistant: Assistant
) -> str:
    """Generate one grounded answer for evaluation."""

    rag_context = rag_service.build_context(
        question
    )

    if rag_context is None:
        return ABSTENTION_MESSAGE

    response_parts = assistant.send_message(
        user_message=question,
        model_message=rag_context.prompt
    )

    return "".join(response_parts).strip()

def evaluate_case(
    case: dict,
    rag_service: RAGService,
    assistant: Assistant
) -> tuple[str, AnswerEvaluation]:
    """Generate and evaluate one independent test case."""

    #Each evaluation question must start without conversation history.
    assistant.reset()

    answer = generate_answer(
        question=case["question"],
        rag_service=rag_service,
        assistant=assistant
    )

    evaluation = evaluate_answer(
        answer=answer,
        expected_phrases=case.get(
            "expected_phrases",
            []
        ),
        forbidden_phrases=case.get(
            "forbidden_phrases",
            []
        ),
        expect_abstention=case.get(
            "expect_abstention",
            False
        )
    )

    return answer, evaluation

def main() -> int:
    """Run the end-to-end answer-grounding evaluation"""

    rag_config = CONFIG.rag

    provider = OllamaProvider(
        model=CONFIG.default_model,
        host=CONFIG.ollama_host
    )

    assistant = Assistant(
        llm=provider,
        system_prompt=SYSTEM_PROMPT
    )

    embedder = OllamaEmbedder(
        model=rag_config.embedding_model,
        query_prefix=rag_config.embedding_query_prefix,
        document_prefix=rag_config.embedding_document_prefix
    )

    vector_store = create_vector_store(
        config=rag_config,
        embedder=embedder
    )

    reranker = create_reranker(
        config=rag_config
    )

    rag_service = RAGService(
        document_directory=rag_config.document_directory,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=rag_config.chunk_size,
        overlap=rag_config.overlap,
        top_k=rag_config.top_k,
        min_score=rag_config.min_score,
        reranker=reranker,
        candidate_k=rag_config.candidate_k
    )

    cases = load_cases()
    passed_cases = 0 

    try:
        rag_service.initialize()

        for case in cases:
            answer, evaluation = evaluate_case(
                case=case,
                rag_service=rag_service,
                assistant=assistant
            )

            status = (
                "PASS"
                if evaluation.passed
                else "FAIL"
            )

            print(
                f"\n[{status}] "
                f"{case['question']}"
            )
            print(f"Answer: {answer}")

            if evaluation.failures:
                print("Failures:")
                for failure in evaluation.failures:
                    print(f"- {failure}")

            if evaluation.passed:
                passed_cases +=1

    finally:
        close_method = getattr(
            vector_store,
            "close",
            None
        )

        if callable(close_method):
            close_method()

        print(
            "\n Answer-grounding evaluation: "
            f"{passed_cases}/{len(cases)} passed"
        )

        return (
            0 
            if passed_cases == len(cases)
            else 1
        )

if __name__ == "__main__":
    raise SystemExit(main())
