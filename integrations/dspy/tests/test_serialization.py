from haystack_integrations.components.generators.dspy.chat.chat_generator import DSPySignatureChatGenerator


class TestSerialization:
    def test_to_dict(self):
        component = DSPySignatureChatGenerator(signature="question -> answer", model="gpt-4o-mini")
        data = component.to_dict()
        base_path = "haystack_integrations.components.generators.dspy.chat.chat_generator"
        expected_path = f"{base_path}.DSPySignatureChatGenerator"
        assert data["type"] == expected_path
        assert data["init_parameters"]["signature"] == {"type": "str", "value": "question -> answer"}

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {"signature": {"type": "str", "value": "question -> answer"}, "model": "gpt-4o-mini"},
        }
        component = DSPySignatureChatGenerator.from_dict(data)
        assert component.model == "gpt-4o-mini"
        assert component.signature == "question -> answer"
