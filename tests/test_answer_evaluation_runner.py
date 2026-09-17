import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.answer_grounding import (
    ABSTENTION_MESSAGE,
    AnswerEvaluation
)
from evaluation.run_answer_evaluation import (
    evaluate_case,
    generate_answer
)


class AnswerEvaluationRunnerTest(unittest.TestCase):

    def test_generate_answer_collects_streamed_grounded_response(self):
        """Join streamed LLM fragments into one answer."""

        rag_service = Mock()
        rag_service.build_context.return_value = SimpleNamespace(
            prompt="Grounded RAG prompt"
        )

        assistant = Mock()
        assistant.send_message.return_value = iter(
            [
                "ORION",
                "-27 is the internal codename."
            ]
        )

        answer = generate_answer(
            question="What is ORION-27?",
            rag_service=rag_service,
            assistant=assistant
        )

        self.assertEqual(
            answer,
            "ORION-27 is the internal codename."
        )
        rag_service.build_context.assert_called_once_with(
            "What is ORION-27?"
        )
        assistant.send_message.assert_called_once_with(
            user_message="What is ORION-27?",
            model_message="Grounded RAG prompt"
        )

    def test_generate_answer_abstains_without_calling_llm(self):
        """Return the exact abstention when retrieval finds no context."""

        rag_service = Mock()
        rag_service.build_context.return_value = None

        assistant = Mock()

        answer = generate_answer(
            question="How do I bake a chocolate cake?",
            rag_service=rag_service,
            assistant=assistant
        )

        self.assertEqual(
            answer,
            ABSTENTION_MESSAGE
        )
        assistant.send_message.assert_not_called()

    @patch(
        "evaluation.run_answer_evaluation.evaluate_answer"
    )
    @patch(
        "evaluation.run_answer_evaluation.generate_answer"
    )
    def test_evaluate_case_applies_case_rules_independently(
        self,
        mock_generate_answer,
        mock_evaluate_answer
    ):
        """Reset memory and apply the expectations from one case."""

        case = {
            "question": "What is ORION-27?",
            "expected_phrases": [
                "ORION-27",
                "internal codename"
            ],
            "expect_abstention": False
        }

        mock_generate_answer.return_value = (
            "ORION-27 is the internal codename."
        )
        expected_evaluation = AnswerEvaluation(
            passed=True,
            failures=[]
        )
        mock_evaluate_answer.return_value = (
            expected_evaluation
        )

        rag_service = Mock()
        assistant = Mock()

        answer, evaluation = evaluate_case(
            case=case,
            rag_service=rag_service,
            assistant=assistant
        )

        assistant.reset.assert_called_once_with()
        mock_generate_answer.assert_called_once_with(
            question=case["question"],
            rag_service=rag_service,
            assistant=assistant
        )
        mock_evaluate_answer.assert_called_once_with(
            answer=answer,
            expected_phrases=case["expected_phrases"],
            expect_abstention=False
        )
        self.assertEqual(
            answer,
            mock_generate_answer.return_value
        )
        self.assertEqual(
            evaluation,
            expected_evaluation
        )


if __name__ == "__main__":
    unittest.main()
