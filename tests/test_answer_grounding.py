import unittest

from evaluation.answer_grounding import (
    ABSTENTION_MESSAGE,
    evaluate_answer
)


class AnswerGroundingEvaluationTest(unittest.TestCase):

    def test_passes_when_answer_contains_all_expected_facts(self):
        evaluation = evaluate_answer(
            answer=(
                "ORION-27 is the internal codename "
                "for the example RAG knowledge base."
            ),
            expected_phrases=[
                "ORION-27",
                "internal codename"
            ]
        )

        self.assertTrue(
            evaluation.passed
        )
        self.assertEqual(
            evaluation.failures,
            []
        )

    def test_ignores_markdown_emphasis_in_expected_facts(self):
        evaluation = evaluate_answer(
            answer=(
                "Eligible employees receive **18** "
                "paid leave days each year."
            ),
            expected_phrases=[
                "18 paid leave days"
            ]
        )

        self.assertTrue(
            evaluation.passed
        )
        self.assertEqual(
            evaluation.failures,
            []
        )

    def test_fails_when_answer_omits_an_expected_fact(self):
        evaluation = evaluate_answer(
            answer="Project Aurora is a local AI assistant.",
            expected_phrases=[
                "local AI assistant",
                "without relying on cloud LLM APIs"
            ]
        )

        self.assertFalse(
            evaluation.passed
        )
        self.assertTrue(
            any(
                "without relying on cloud llm apis"
                in failure.lower()
                for failure in evaluation.failures
            )
        )

    def test_fails_when_answer_contains_a_forbidden_fact(self):
        evaluation = evaluate_answer(
            answer=(
                "ORION-27 is the internal codename. "
                "It is also the Project Aurora assistant."
            ),
            expected_phrases=[
                "ORION-27",
                "internal codename"
            ],
            forbidden_phrases=[
                "Project Aurora"
            ]
        )

        self.assertFalse(
            evaluation.passed
        )
        self.assertTrue(
            any(
                "forbidden phrase"
                in failure.lower()
                and "project aurora"
                in failure.lower()
                for failure in evaluation.failures
            )
        )

    def test_fails_when_answer_contains_a_source_reference(self):
        evaluation = evaluate_answer(
            answer=(
                "ORION-27 is the internal codename. "
                "See https://example.com/rag_test_notes.txt."
            ),
            expected_phrases=[
                "ORION-27",
                "internal codename"
            ]
        )

        self.assertFalse(
            evaluation.passed
        )
        self.assertTrue(
            any(
                "source reference"
                in failure.lower()
                for failure in evaluation.failures
            )
        )

    def test_requires_exact_abstention_for_unsupported_questions(self):
        passing_evaluation = evaluate_answer(
            answer=ABSTENTION_MESSAGE,
            expected_phrases=[],
            expect_abstention=True
        )

        failing_evaluation = evaluate_answer(
            answer=(
                "I do not know how to bake "
                "a chocolate cake."
            ),
            expected_phrases=[],
            expect_abstention=True
        )

        self.assertTrue(
            passing_evaluation.passed
        )
        self.assertFalse(
            failing_evaluation.passed
        )
        self.assertTrue(
            any(
                "exact abstention"
                in failure.lower()
                for failure in failing_evaluation.failures
            )
        )


if __name__ == "__main__":
    unittest.main()
