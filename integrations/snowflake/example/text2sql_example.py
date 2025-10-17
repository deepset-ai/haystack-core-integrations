from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from haystack_integrations.components.retrievers.snowflake import SnowflakeTableRetriever

load_dotenv()

sql_template = """
    You are a SQL expert working with Snowflake.

    Your task is to create a Snowflake SQL query for the given question.

    Refrain from explaining your answer. Your answer must be the SQL query
    in plain text format without using Markdown.

    Here are some relevant tables, a description about it, and their
    columns:

    Database name: DEMO_DB
    Schema name: ADVENTURE_WORKS
    Table names:
    - ADDRESS: Employees Address Table
    - EMPLOYEE: Employees directory
    - SALESTERRITORY: Sales territory lookup table.
    - SALESORDERHEADER: General sales order information.

    User's question: {{ question }}

    Generated SQL query:
"""

sql_builder = PromptBuilder(template=sql_template)

analyst_template = """
    You are an expert data analyst.

    Your role is to answer the user's question {{ question }} using the information
    in the table.

    You will base your response solely on the information provided in the
    table.

    Do not rely on your knowledge base; only the data that is in the table.

    Refrain from using the term "table" in your response, but instead, use
    the word "data"

    If the table is blank say:

    "The specific answer can't be found in the database. Try rephrasing your
    question."

    Additionally, you will present the table in a tabular format and provide
    the SQL query used to extract the relevant rows from the database in
    Markdown.

    If the table is larger than 10 rows, display the most important rows up
    to 10 rows. Your answer must be detailed and provide insights based on
    the question and the available data.

    SQL query:

    {{ sql_query }}

    Table:

    {{ table }}

    Answer:
"""

analyst_builder = PromptBuilder(template=analyst_template)

# LLM responsible for generating the SQL query
sql_llm = OpenAIGenerator(
    model="gpt-4o",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    generation_kwargs={"temperature": 0.0, "max_tokens": 1000},
)

# LLM responsible for analyzing the table
analyst_llm = OpenAIGenerator(
    model="gpt-4o",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    generation_kwargs={"temperature": 0.0, "max_tokens": 2000},
)

snowflake = SnowflakeTableRetriever(
    user="<ACCOUNT-USER>",
    account="<ACCOUNT-IDENTIFIER>",
    authenticator="SNOWFLAKE",
    api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
    warehouse="<WAREHOUSE-NAME>",
)

adapter = OutputAdapter(template="{{ replies[0] }}", output_type=str)

pipeline = Pipeline()

pipeline.add_component(name="sql_builder", instance=sql_builder)
pipeline.add_component(name="sql_llm", instance=sql_llm)
pipeline.add_component(name="adapter", instance=adapter)
pipeline.add_component(name="snowflake", instance=snowflake)
pipeline.add_component(name="analyst_builder", instance=analyst_builder)
pipeline.add_component(name="analyst_llm", instance=analyst_llm)


pipeline.connect("sql_builder.prompt", "sql_llm.prompt")
pipeline.connect("sql_llm.replies", "adapter.replies")
pipeline.connect("adapter.output", "snowflake.query")
pipeline.connect("snowflake.table", "analyst_builder.table")
pipeline.connect("adapter.output", "analyst_builder.sql_query")
pipeline.connect("analyst_builder.prompt", "analyst_llm.prompt")

question = "What are my top territories by number of orders and by sales value?"

response = pipeline.run(data={"sql_builder": {"question": question}, "analyst_builder": {"question": question}})
