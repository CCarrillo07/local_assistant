import unittest
from unittest.mock import patch

import main
from config import AppConfig, RAGConfig


class MainRAGModeTest(unittest.TestCase):
    
    @patch(
        "builtins.input",
        side_effect=["/exit"]
    )
    @patch("main.RAGService")
    @patch("main.OllamaEmbedder")
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_required_mode_initializes_rag_at_startup(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_embedder_class,
        mock_rag_service_class,
        mock_input
    ):
        test_config = AppConfig(
            default_model="test-model",
            rag=RAGConfig(
                available=True,
                mode="required",
                allow_user_control=False,
                vector_store_backend="npz"
            )
        )
        
        mock_rag_service = (
            mock_rag_service_class.return_value
        )
        
        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()
            
        mock_rag_service_class.assert_called_once()
        mock_rag_service.initialize.assert_called_once_with()
        
    @patch(
        "builtins.input",
        side_effect=[
            "/rag off",
            "What is ORION-27?",
            "exit"
        ]
    )
    @patch("builtins.print")
    @patch("main.RAGService")
    @patch("main.OllamaEmbedder")
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_required_mode_rejects_control_and_keeps_rag_enabled(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_embedder_class,
        mock_rag_service_class,
        mock_print,
        mock_input
    ):
        test_config = AppConfig(
            default_model="test-model",
            rag=RAGConfig(
                available=True,
                mode="required",
                allow_user_control=False,
                vector_store_backend="npz"
            )
        )
        
        mock_rag_service = (
            mock_rag_service_class.return_value
        )
        mock_rag_service.build_prompt.return_value = (
            "Augmented RAG prompt"
        )
        
        mock_assistant = mock_assistant_class.return_value
        mock_assistant.send_message.return_value = iter(
            ["Grounded answer"]
        )
        
        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()
            
        mock_print.assert_any_call(
            "\nRAG controls are not available "
            "in this deployment."
        )
        
        mock_rag_service.build_prompt.assert_called_once_with(
            "What is ORION-27?"
        )
        
        mock_assistant.send_message.assert_called_once_with(
            user_message="What is ORION-27?",
            model_message="Augmented RAG prompt"
        )
        
    @patch(
        "builtins.input",
        side_effect=[
            "How do I bake a chocolate cake?",
            "/exit"
        ]
    )
    @patch("builtins.print")
    @patch("main.RAGService")
    @patch("main.OllamaEmbedder")
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_required_mode_abstains_without_calling_llm(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_embedder_class,
        mock_rag_service_class,
        mock_print,
        mock_input
    ):
        test_config = AppConfig(
            default_model="test-model",
            rag=RAGConfig(
                available=True,
                mode="required",
                allow_user_control=False,
                vector_store_backend="npz"
            )
        )
        
        mock_rag_service = (
            mock_rag_service_class.return_value
        )
        
        mock_rag_service.build_prompt.return_value = None

        mock_assistant = mock_assistant_class.return_value
        
        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()
            
        mock_rag_service.build_prompt.assert_called_once_with(
            "How do I bake a chocolate cake?"
        )
        
        mock_assistant.send_message.assert_not_called()
        
        mock_print.assert_any_call(
            "The answer could not be found "
            "in the documents."
        )

    @patch(
        "builtins.input",
        side_effect=[
            "How do I bake a chocolate cake?",
            "/exit"
        ]
    )
    @patch("main.RAGService")
    @patch("main.OllamaEmbedder")
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_auto_mode_uses_general_assistant_without_context(
        self, 
        mock_provider_class,
        mock_assistant_class,
        mock_embedder_class,
        mock_rag_service_class,
        mock_input
    ):
        test_config = AppConfig(
            default_model="test-model",
            rag=RAGConfig(
                available=True,
                mode="auto",
                allow_user_control=False,
                vector_store_backend="npz"
            )
        )

        mock_rag_service = (
            mock_rag_service_class.return_value
        )
        mock_rag_service.build_prompt.return_value = None

        mock_assistant = mock_assistant_class.return_value
        mock_assistant.send_message.return_value = iter(
            ["General answer"]
        )

        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()

        mock_rag_service.initialize.assert_called_once_with()

        mock_rag_service.build_prompt.assert_called_once_with(
            "How do I bake a chocolate cake?"
        )

        mock_assistant.send_message.assert_called_once_with(
            user_message="How do I bake a chocolate cake?",
            model_message=None
        )

    @patch(
        "builtins.input",
        side_effect=["/exit"]
    )
    @patch("main.RAGService")
    @patch("main.create_reranker")
    @patch("main.create_vector_store")
    @patch("main.OllamaEmbedder")
    @patch("main.Assistant")
    @patch("main.OllamaProvider")
    def test_wires_configured_reranker_into_rag_service(
        self,
        mock_provider_class,
        mock_assistant_class,
        mock_embedder_class,
        mock_vector_store_factory,
        mock_reranker_factory,
        mock_rag_service_class,
        mock_input
    ):
        test_config = AppConfig(
            default_model="test-model",
            rag=RAGConfig(
                available=True,
                mode="manual",
                allow_user_control=True,
                vector_store_backend="npz",
                reranker_enabled=True,
                candidate_k=5
            )
        )

        mock_embedder = (
            mock_embedder_class.return_value
        )
        mock_vector_store = (
            mock_vector_store_factory.return_value
        )
        mock_reranker = (
            mock_reranker_factory.return_value
        )

        with patch.object(
            main,
            "CONFIG",
            test_config
        ):
            main.main()

        mock_reranker_factory.assert_called_once_with(
            config=test_config.rag
        )

        mock_rag_service_class.assert_called_once_with(
            document_directory=(
                test_config.rag.document_directory
            ),
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            chunk_size=test_config.rag.chunk_size,
            overlap=test_config.rag.overlap,
            top_k=test_config.rag.top_k,
            min_score=test_config.rag.min_score,
            reranker=mock_reranker,
            candidate_k=test_config.rag.candidate_k
        )
    
        
if __name__ == "__main__":
    unittest.main()
