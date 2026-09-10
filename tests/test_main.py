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
                allow_user_control=False
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
        
if __name__ == "__main__":
    unittest.main()