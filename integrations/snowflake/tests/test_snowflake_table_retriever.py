# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dateutil.tz import tzlocal
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pytest import LogCaptureFixture
from snowflake.connector.errors import DatabaseError, ForbiddenError, ProgrammingError

from haystack_integrations.components.retrievers.snowflake import SnowflakeTableRetriever


class TestSnowflakeTableRetriever:
    @pytest.fixture
    def snowflake_table_retriever(self) -> SnowflakeTableRetriever:
        return SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            api_key=Secret.from_token("test-api-key"),
            database="test_database",
            db_schema="test_schema",
            warehouse="test_warehouse",
            login_timeout=30,
        )

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_snowflake_connector(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        conn = snowflake_table_retriever._snowflake_connector(
            connect_params={
                "user": "test_user",
                "account": "test_account",
                "api_key": Secret.from_token("test-api-key"),
                "database": "test_database",
                "schema": "test_schema",
                "warehouse": "test_warehouse",
                "login_timeout": 30,
            }
        )
        mock_connect.assert_called_once_with(
            user="test_user",
            account="test_account",
            api_key=Secret.from_token("test-api-key"),
            database="test_database",
            schema="test_schema",
            warehouse="test_warehouse",
            login_timeout=30,
        )

        assert conn == mock_conn

    def test_query_is_empty(
        self, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        query = ""
        result = snowflake_table_retriever.run(query=query)

        assert result["table"] == ""
        assert result["dataframe"].empty
        assert "Provide a valid SQL query" in caplog.text

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_exception(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_connect = mock_connect.return_value
        mock_connect._fetch_data.side_effect = Exception("Unknown error")

        query = 4
        result = snowflake_table_retriever.run(query=query)

        assert result["table"] == ""
        assert result["dataframe"].empty

        assert "An unexpected error has occurred" in caplog.text

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_forbidden_error_during_connection(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_connect.side_effect = ForbiddenError(msg="Forbidden error", errno=403)

        result = snowflake_table_retriever._fetch_data(query="SELECT * FROM test_table")

        assert result.empty
        assert "000403: Forbidden error" in caplog.text

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_programing_error_during_connection(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_connect.side_effect = ProgrammingError(msg="Programming error", errno=403)

        result = snowflake_table_retriever._fetch_data(query="SELECT * FROM test_table")

        assert result.empty
        assert "000403: Programming error" in caplog.text

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_execute_sql_query_programming_error(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value

        mock_cursor.execute.side_effect = ProgrammingError(msg="Simulated programming error", sfqid="ABC-123")

        result = snowflake_table_retriever._execute_sql_query(mock_conn, "SELECT * FROM some_table")

        assert result.empty

        assert (
            "Simulated programming error Use the following ID to check the status of "
            "the query in Snowflake UI (ID: ABC-123)" in caplog.text
        )

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_run_connection_error(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        mock_connect.side_effect = DatabaseError(msg="Connection error", errno=1234)

        query = "SELECT * FROM test_table"
        result = snowflake_table_retriever.run(query=query)

        assert result["table"] == ""
        assert result["dataframe"].empty

    def test_extract_single_table_name(self, snowflake_table_retriever: SnowflakeTableRetriever) -> None:
        queries = [
            "SELECT * FROM table_a",
            "SELECT name, value FROM (SELECT name, value FROM table_a) AS subquery",
            "SELECT name, value FROM (SELECT name, value FROM table_a ) AS subquery",
            "UPDATE table_a SET value = 'new_value' WHERE id = 1",
            "INSERT INTO table_a (id, name, value) VALUES (1, 'name1', 'value1')",
            "DELETE FROM table_a WHERE id = 1",
            "TRUNCATE TABLE table_a",
            "DROP TABLE table_a",
        ]
        for query in queries:
            result = snowflake_table_retriever._extract_table_names(query)
            assert result == ["TABLE_A"]

    def test_extract_database_and_schema_from_query(self, snowflake_table_retriever: SnowflakeTableRetriever) -> None:
        # when database and schema are next to table name
        assert snowflake_table_retriever._extract_table_names(query="SELECT * FROM DB.SCHEMA.TABLE_A") == [
            "DB.SCHEMA.TABLE_A"
        ]
        # No database
        assert snowflake_table_retriever._extract_table_names(query="SELECT * FROM SCHEMA.TABLE_A") == [
            "SCHEMA.TABLE_A"
        ]

    def test_extract_multiple_table_names(self, snowflake_table_retriever: SnowflakeTableRetriever) -> None:
        queries = [
            "MERGE INTO table_a USING table_b ON table_a.id = table_b.id WHEN MATCHED",
            "SELECT a.name, b.value FROM table_a AS a FULL OUTER JOIN table_b AS b ON a.id = b.id",
            "SELECT a.name, b.value FROM table_a AS a RIGHT JOIN table_b AS b ON a.id = b.id",
        ]
        for query in queries:
            result = snowflake_table_retriever._extract_table_names(query)
            # Due to using set when deduplicating
            assert result == ["TABLE_A", "TABLE_B"] or ["TABLE_B", "TABLE_A"]

    def test_extract_multiple_db_schema_from_table_names(
        self, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        assert (
            snowflake_table_retriever._extract_table_names(
                query="""SELECT a.name, b.value FROM DB.SCHEMA.TABLE_A AS a
                 FULL OUTER JOIN DATABASE.SCHEMA.TABLE_b AS b ON a.id = b.id"""
            )
            == ["DB.SCHEMA.TABLE_A", "DATABASE.SCHEMA.TABLE_A"]
            or ["DATABASE.SCHEMA.TABLE_A", "DB.SCHEMA.TABLE_B"]
        )
        # No database
        assert (
            snowflake_table_retriever._extract_table_names(
                query="""SELECT a.name, b.value FROM SCHEMA.TABLE_A AS a
                 FULL OUTER JOIN SCHEMA.TABLE_b AS b ON a.id = b.id"""
            )
            == ["SCHEMA.TABLE_A", "SCHEMA.TABLE_A"]
            or ["SCHEMA.TABLE_A", "SCHEMA.TABLE_B"]
        )

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_execute_sql_query(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col1.name = "City"
        mock_col2.name = "State"
        mock_cursor.fetchmany.return_value = [("Chicago", "Illinois")]
        mock_cursor.description = [mock_col1, mock_col2]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        query = "SELECT * FROM test_table"
        expected = pd.DataFrame(data={"City": ["Chicago"], "State": ["Illinois"]})
        result = snowflake_table_retriever._execute_sql_query(conn=mock_conn, query=query)

        mock_cursor.execute.assert_called_once_with(query)

        assert result.equals(expected)

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_is_select_only(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_cursor.fetchall.side_effect = [
            [("DATETIME", "ROLE_NAME", "USER", "USER_NAME", "GRANTED_BY")],  # User roles
            [
                (
                    "DATETIME",
                    "SELECT",
                    "TABLE",
                    "LOCATIONS",
                    "ROLE",
                    "ROLE_NAME",
                    "GRANT_OPTION",
                    "GRANTED_BY",
                )
            ],  # Table privileges
        ]

        query = "select * from locations"
        result = snowflake_table_retriever._check_privilege(conn=mock_conn, user="test_user", query=query)
        assert result

        mock_cursor.fetchall.side_effect = [
            [("DATETIME", "ROLE_NAME", "USER", "USER_NAME", "GRANTED_BY")],  # User roles
            [
                (
                    "DATETIME",
                    "INSERT",
                    "TABLE",
                    "LOCATIONS",
                    "ROLE",
                    "ROLE_NAME",
                    "GRANT_OPTION",
                    "GRANTED_BY",
                )
            ],
        ]

        result = snowflake_table_retriever._check_privilege(conn=mock_conn, user="test_user", query=query)

        assert not result

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_column_after_from(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col1.name = "id"
        mock_col2.name = "year"
        mock_cursor.fetchmany.return_value = [(1233, 1998)]
        mock_cursor.description = [mock_col1, mock_col2]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        query = "SELECT id, extract(year from date_col) as year FROM test_table"
        expected = pd.DataFrame(data={"id": [1233], "year": [1998]})
        result = snowflake_table_retriever._execute_sql_query(conn=mock_conn, query=query)
        mock_cursor.execute.assert_called_once_with(query)

        assert result.equals(expected)

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_run(self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("DATETIME", "ROLE_NAME", "USER", "USER_NAME", "GRANTED_BY")],  # User roles
            [
                (
                    "DATETIME",
                    "SELECT",
                    "TABLE",
                    "locations",
                    "ROLE",
                    "ROLE_NAME",
                    "GRANT_OPTION",
                    "GRANTED_BY",
                )
            ],
        ]
        mock_col1.name = "City"
        mock_col2.name = "State"
        mock_cursor.description = [mock_col1, mock_col2]

        mock_cursor.fetchmany.return_value = [("Chicago", "Illinois")]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        query = "SELECT * FROM locations"

        expected = {
            "dataframe": pd.DataFrame(data={"City": ["Chicago"], "State": ["Illinois"]}),
            "table": "| City    | State    |\n|:--------|:---------|\n| Chicago | Illinois |",
        }

        result = snowflake_table_retriever.run(query=query)

        assert result["dataframe"].equals(expected["dataframe"])
        assert result["table"] == expected["table"]

    @pytest.fixture
    def mock_chat_completion(self) -> Generator:
        """
        Mock the OpenAI API completion response and reuse it for tests
        """
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            completion = ChatCompletion(
                id="foo",
                model="gpt-4o-mini",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="stop",
                        logprobs=None,
                        index=0,
                        message=ChatCompletionMessage(content="select locations from table_a", role="assistant"),
                    )
                ],
                created=int(datetime.now(tz=tzlocal()).timestamp()),
                usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
            )

            mock_chat_completion_create.return_value = completion
            yield mock_chat_completion_create

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_run_pipeline(
        self,
        mock_connect: MagicMock,
        mock_chat_completion: MagicMock,
        snowflake_table_retriever: SnowflakeTableRetriever,
    ) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_col1 = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("DATETIME", "ROLE_NAME", "USER", "USER_NAME", "GRANTED_BY")],  # User roles
            [
                (
                    "DATETIME",
                    "SELECT",
                    "TABLE",
                    "test_database.test_schema.table_a",
                    "ROLE",
                    "ROLE_NAME",
                    "GRANT_OPTION",
                    "GRANTED_BY",
                )
            ],
        ]
        mock_col1.name = "locations"

        mock_cursor.description = [mock_col1]

        mock_cursor.fetchmany.return_value = [("Chicago",), ("Miami",), ("Berlin",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        expected = {
            "dataframe": pd.DataFrame(data={"locations": ["Chicago", "Miami", "Berlin"]}),
            "table": "| locations   |\n|:------------|\n| Chicago     |\n| Miami       |\n| Berlin      |",
        }

        llm = OpenAIGenerator(model="gpt-4o-mini", api_key=Secret.from_token("test-api-key"))
        adapter = OutputAdapter(template="{{ replies[0] }}", output_type=str)
        pipeline = Pipeline()

        pipeline.add_component("llm", llm)
        pipeline.add_component("adapter", adapter)
        pipeline.add_component("snowflake", snowflake_table_retriever)

        pipeline.connect(sender="llm.replies", receiver="adapter.replies")
        pipeline.connect(sender="adapter.output", receiver="snowflake.query")

        result = pipeline.run(data={"llm": {"prompt": "Generate a SQL query that extract all locations from table_a"}})

        assert result["snowflake"]["dataframe"].equals(expected["dataframe"])
        assert result["snowflake"]["table"] == expected["table"]

    def test_from_dict(self, monkeypatch: MagicMock) -> None:
        monkeypatch.setenv("SNOWFLAKE_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever"
            ".SnowflakeTableRetriever",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["SNOWFLAKE_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "user": "test_user",
                "account": "new_account",
                "database": "test_database",
                "db_schema": "test_schema",
                "warehouse": "test_warehouse",
                "login_timeout": 3,
            },
        }
        component = SnowflakeTableRetriever.from_dict(data)

        assert component.user == "test_user"
        assert component.account == "new_account"
        assert component.api_key == Secret.from_env_var("SNOWFLAKE_API_KEY")
        assert component.database == "test_database"
        assert component.db_schema == "test_schema"
        assert component.warehouse == "test_warehouse"
        assert component.login_timeout == 3

    def test_to_dict_default(self, monkeypatch: MagicMock) -> None:
        monkeypatch.setenv("SNOWFLAKE_API_KEY", "test-api-key")
        component = SnowflakeTableRetriever(
            user="test_user",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            account="test_account",
            database="test_database",
            db_schema="test_schema",
            warehouse="test_warehouse",
            login_timeout=30,
        )

        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["SNOWFLAKE_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "user": "test_user",
                "account": "test_account",
                "database": "test_database",
                "db_schema": "test_schema",
                "warehouse": "test_warehouse",
                "login_timeout": 30,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch: MagicMock) -> None:
        monkeypatch.setenv("SNOWFLAKE_API_KEY", "test-api-key")
        monkeypatch.setenv("SNOWFLAKE_API_KEY", "test-api-key")
        component = SnowflakeTableRetriever(
            user="John",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            account="TGMD-EEREW",
            database="CITY",
            db_schema="SMALL_TOWNS",
            warehouse="COMPUTE_WH",
            login_timeout=30,
        )

        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["SNOWFLAKE_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "user": "John",
                "account": "TGMD-EEREW",
                "database": "CITY",
                "db_schema": "SMALL_TOWNS",
                "warehouse": "COMPUTE_WH",
                "login_timeout": 30,
            },
        }

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_has_select_privilege(
        self, mock_logger: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever
    ) -> None:
        # Define test cases
        test_cases = [
            # Test case 1: Fully qualified table name in query
            {
                "privileges": [[None, "SELECT", None, "table"]],
                "table_name": "table",
                "expected_result": True,
            },
            # Test case 2: Schema and table names in query, database name as argument
            {
                "privileges": [[None, "SELECT", None, "table"]],
                "table_name": "table",
                "expected_result": True,
            },
            # Test case 3: Only table name in query, database and schema names as arguments
            {
                "privileges": [[None, "SELECT", None, "table"]],
                "table_name": "table",
                "expected_result": True,
            },
            # Test case 5: Privilege does not match
            {
                "privileges": [[None, "INSERT", None, "table"]],
                "table_name": "table",
                "expected_result": False,
            },
            # Test case 6: Case-insensitive match
            {
                "privileges": [[None, "select", None, "table"]],
                "table_name": "TABLE",
                "expected_result": True,
            },
        ]

        for case in test_cases:
            result = snowflake_table_retriever._has_select_privilege(
                privileges=case["privileges"],  # type: ignore
                table_name=case["table_name"],  # type: ignore
            )
            assert result == case["expected_result"]  # type: ignore

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_user_does_not_exist(
        self, mock_connect: MagicMock, snowflake_table_retriever: SnowflakeTableRetriever, caplog: LogCaptureFixture
    ) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.fetchall.return_value = []

        result = snowflake_table_retriever._fetch_data(query="""SELECT * FROM test_table""")

        assert result.empty
        assert "User does not exist" in caplog.text

    @patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.snowflake.connector.connect"
    )
    def test_empty_query(self, snowflake_table_retriever: SnowflakeTableRetriever) -> None:
        result = snowflake_table_retriever._fetch_data(query="")

        assert result.empty

    def test_serialization_deserialization_pipeline(self) -> None:

        pipeline = Pipeline()
        pipeline.add_component("snow", SnowflakeTableRetriever(user="test_user", account="test_account"))
        pipeline.add_component("prompt_builder", PromptBuilder(template="Display results {{ table }}"))
        pipeline.connect("snow.table", "prompt_builder.table")

        pipeline_dict = pipeline.to_dict()

        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline == pipeline
