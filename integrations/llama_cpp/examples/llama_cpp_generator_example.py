from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

generator = LlamaCppGenerator(model="openchat-3.5-1210.Q3_K_S.gguf", n_ctx=512, n_batch=128)
generator.warm_up()

question = "Who is the best American actor?"
prompt = f"GPT4 Correct User: {question} <|end_of_turn|> GPT4 Correct Assistant:"

result = generator.run(prompt, generation_kwargs={"max_tokens": 128})
generated_text = result["replies"][0]

print(generated_text)
