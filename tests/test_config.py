import unittest

from config import RAGConfig

class RAGConfigTest(unittest.TestCase):

    def test_accepts_valid_rag_configurations(self):

        valid_configurations = [
            {
                "available": True,
                "mode": "manual",
                "allow_user_control": True
            },
            {
                "available": True,
                "mode": "auto",
                "allow_user_control": False
            },
            {
                "available": True,
                "mode": "required",
                "allow_user_control": False
            },
            {
                "available": False,
                "mode": "manual",
                "allow_user_control": False
            }
        ]

        for configuration in valid_configurations:

            with self.subTest(
                configuration=configuration
            ):
                rag_config = RAGConfig(
                    **configuration
                )

                self.assertEqual(
                    rag_config.available,
                    configuration["available"]
                )

                self.assertEqual(
                    rag_config.mode,
                    configuration["mode"]
                )

                self.assertEqual(
                    rag_config.allow_user_control,
                    configuration["allow_user_control"]
                )

    def test_rejects_invalid_user_control_combinations(self):

        invalid_configurations = [
            {
                "available": True,
                "mode": "manual",
                "allow_user_control": False
            },
            {
                "available": True,
                "mode": "auto",
                "allow_user_control": True
            },
            {
                "available": True,
                "mode": "required",
                "allow_user_control": True
            },
            {
                "available": False,
                "mode": "manual",
                "allow_user_control": True
            },
        ]

        for configuration in invalid_configurations:

            with self.subTest(
                configuration=configuration
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "allow_user_control"
                ):
                    RAGConfig(
                        **configuration
                    )

    def test_rejects_invalid_rag_mode(self):

        with self.assertRaisesRegex(
            ValueError,
            "Invalid RAG mode"
        ):
            RAGConfig(
            available=True,
            mode="automatic",
            allow_user_control=False
            )

if __name__ == "__main__":
    unittest.main()