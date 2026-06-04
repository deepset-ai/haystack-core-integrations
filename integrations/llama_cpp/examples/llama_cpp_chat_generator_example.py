from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator

generator = LlamaCppChatGenerator(model="openchat-3.5-1210.Q3_K_S.gguf", n_ctx=512, n_batch=128)
# Components warm up automatically on first run.

question = "Who is the best American actor?"
result = generator.run(question, generation_kwargs={"max_tokens": 128})
print(result["replies"][0].text)
