from assistant import Assistant
from evaluation.answer_grounding import(
    ABSTENTION_MESSAGE,
    AnswerEvaluation,
    evaluate_answer
)
from rag.service import RAGService

def generate_answer(
    question: str,
    rag_service: RAGService,
    assistant: Assistant
) -> str:
    """Generate one grounder answer for evaluation."""

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
        expect_abstention=case.get(
            "expect_abstention",
            False
        )
    )

    return answer, evaluation
           