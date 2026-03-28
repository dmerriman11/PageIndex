import unittest

from pageindex.client import PageIndexClient


class RetrieveModelNormalizationTests(unittest.TestCase):
    def test_plain_model_name_is_unchanged(self):
        client = PageIndexClient(retrieve_model="gpt-5.4")
        self.assertEqual(client.retrieve_model, "gpt-5.4")

    def test_litellm_prefix_is_preserved(self):
        client = PageIndexClient(retrieve_model="litellm/anthropic/claude-3-5-sonnet")
        self.assertEqual(client.retrieve_model, "litellm/anthropic/claude-3-5-sonnet")

    def test_openai_prefix_is_preserved(self):
        client = PageIndexClient(retrieve_model="openai/gpt-4.1")
        self.assertEqual(client.retrieve_model, "openai/gpt-4.1")

    def test_any_llm_prefix_is_routed_via_litellm(self):
        client = PageIndexClient(retrieve_model="any-llm/gpt-4.1")
        self.assertEqual(client.retrieve_model, "litellm/any-llm/gpt-4.1")

    def test_other_provider_paths_are_routed_via_litellm(self):
        client = PageIndexClient(retrieve_model="anthropic/claude-3-5-sonnet")
        self.assertEqual(client.retrieve_model, "litellm/anthropic/claude-3-5-sonnet")


if __name__ == "__main__":
    unittest.main()
