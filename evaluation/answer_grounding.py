from dataclasses import dataclass

ABSTENTION_MESSAGE = (
    "The answer could not be found "
    "in the documents."
)

FORBIDDEN_SOURCE_PATTERNS = (
    "http://",
    "https://",
    "www.",
    "[source",
    "source 1: ",
    ".txt",
    ".pdf"
)

@dataclass(frozen=True)
class AnswerEvaluation:
    """Result of evaluating one generated answer."""

    passed: bool
    failures: list[str]

def normalize_text(
    text: str
) -> str: 
    """Normalize capitalization and whitespace."""

    return " ".join(
        text.lower().split()
    )

def evaluate_answer(
    answer: str,
    expected_phrases: list[str],
    expect_abstention: bool = False
) -> AnswerEvaluation:
    """Evaluate facts, abstention, and source-reference rules."""

    failures = []
    normalized_answer = normalize_text(
        answer
    )

    if expect_abstention:

        if normalized_answer != normalize_text(
            ABSTENTION_MESSAGE
        ):
            failures.append(
                "Answer did not use the exact "
                "abstention message."
            )

        return AnswerEvaluation(
            passed =not failures,
            failures=failures
        )

    for expected_phrase in expected_phrases:

        normalized_phrase = normalize_text(
            expected_phrase
        )

        if normalized_phrase not in normalized_answer:
            failures.append(
                "Missing expected phrase; "
                f"{expected_phrase}"
            )

    for pattern in FORBIDDEN_SOURCE_PATTERNS:

        if pattern in normalized_answer:
            failures.append(
                "Answer contains a forbidden "
                "source reference"
            )
            break

    return AnswerEvaluation(
        passed=not failures,
        failures=failures
    )
