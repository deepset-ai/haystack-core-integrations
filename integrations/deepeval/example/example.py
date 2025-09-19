# A valid OpenAI API key is required to run this example.

from haystack import Pipeline

from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric

QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, "
        "number of participants, and economic impact.",
        "Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports "
        "personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people.",
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language.",
        "Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write "
        "clear, logical code for both small and large-scale software projects.",
    ],
]
RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]

pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.FAITHFULNESS,
    metric_params={"model": "gpt-4"},
)
pipeline.add_component("evaluator", evaluator)

# Each metric expects a specific set of parameters as input. Refer to the
# DeepEvalMetric class' documentation for more details.
results = pipeline.run({"evaluator": {"questions": QUESTIONS, "contexts": CONTEXTS, "responses": RESPONSES}})

for output in results["evaluator"]["results"]:
    print(output)  # noqa: T201
