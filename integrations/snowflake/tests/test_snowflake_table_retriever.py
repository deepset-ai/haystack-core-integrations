import os
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest
from haystack import logging
from haystack.utils import Secret

from haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever import (
    SnowflakeTableRetriever,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def retriever(mocker: Mock) -> SnowflakeTableRetriever:
    mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})
    return SnowflakeTableRetriever(
        user="test_user",
        account="test_account",
        api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
        database="test_db",
        db_schema="test_schema",
        warehouse="test_warehouse",
        return_markdown=True,
    )


@pytest.fixture
def empty_response() -> Dict[str, Any]:
    return SnowflakeTableRetriever._empty_response()


@pytest.fixture
def toy_polars_df() -> pl.DataFrame:
    return pl.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})


@pytest.fixture
def toy_pandas_df(toy_polars_df: pl.DataFrame) -> pd.DataFrame:
    return toy_polars_df.to_pandas()


@pytest.fixture
def expected_markdown() -> str:
    return "shape: (3, 2)\n| col1 | col2 |\n|------|------|\n| 1    | A    |\n| 2    | B    |\n| 3    | C    |"


class TestSnowflakeTableRetriever:
    def test_init_and_serialization(self, retriever: SnowflakeTableRetriever) -> None:
        serialized = retriever.to_dict()
        init_params = serialized["init_parameters"]

        assert "init_parameters" in serialized
        assert init_params["user"] == "test_user"
        assert init_params["account"] == "test_account"
        assert init_params["return_markdown"] is True

        deserialized = SnowflakeTableRetriever.from_dict(serialized)
        assert isinstance(deserialized, SnowflakeTableRetriever)
        assert deserialized.user == "test_user"
        assert deserialized.account == "test_account"
        assert deserialized.return_markdown is True

    @pytest.mark.parametrize(
        "user, account, db_name, schema_name, warehouse_name, expected_uri, should_raise",
        [
            (
                "test_user",
                "test_account",
                "test_db",
                "test_schema",
                "test_warehouse",
                "snowflake://test_user:test_api_key@test_account/test_db/test_schema?warehouse=test_warehouse&login_timeout=60",
                False,
            ),
            (
                "test_user",
                "test_account",
                "test_db",
                None,
                "test_warehouse",
                "snowflake://test_user:test_api_key@test_account/test_db?warehouse=test_warehouse&login_timeout=60",
                False,
            ),
            (
                "test_user",
                "test_account",
                None,
                None,
                None,
                "snowflake://test_user:test_api_key@test_account?login_timeout=60",
                False,
            ),
            (
                None,
                "test_account",
                "test_db",
                "test_schema",
                "test_warehouse",
                None,
                True,
            ),
            ("test_user", None, "test_db", "test_schema", "test_warehouse", None, True),
            (None, None, "test_db", "test_schema", "test_warehouse", None, True),
        ],
    )
    def test_snowflake_uri_constructor(
        self,
        mocker: Mock,
        user: str,
        account: str,
        db_name: Optional[str],
        schema_name: Optional[str],
        warehouse_name: Optional[str],
        expected_uri: Optional[str],
        should_raise: bool,
    ) -> None:
        mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})

        retriever = SnowflakeTableRetriever(
            user=user,
            account=account,
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            database=db_name,
            db_schema=schema_name,
            warehouse=warehouse_name,
        )

        if should_raise:
            with pytest.raises(
                ValueError,
                match="Missing required Snowflake connection parameters: user and account.",
            ):
                retriever._snowflake_uri_constructor()
        else:
            assert retriever._snowflake_uri_constructor() == expected_uri

    @pytest.mark.parametrize(
        "exception, expected_log_level, expected_log_msg",
        [
            (
                RuntimeError("SQL compilation error: invalid identifier 'TEST_COLUMN'"),
                "WARNING",
                "SQL compilation error encountered: SQL compilation error: invalid identifier 'TEST_COLUMN'",
            ),
            (
                ValueError("SQL compilation error: syntax error near 'SELECT'"),
                "WARNING",
                "SQL compilation error encountered: SQL compilation error: syntax error near 'SELECT'",
            ),
            (
                RuntimeError("ADBC connection failed"),
                "ERROR",
                "Error executing query via ADBC - Error N/A: ADBC connection failed",
            ),
            (
                ValueError("Unexpected database error"),
                "ERROR",
                "Error executing query via ADBC - Error N/A: Unexpected database error",
            ),
        ],
    )
    def test_run_sql_error_handling(
        self,
        retriever: SnowflakeTableRetriever,
        caplog: pytest.LogCaptureFixture,
        mocker: Mock,
        exception: Exception,
        expected_log_level: str,
        expected_log_msg: str,
        empty_response: Dict[str, Any],
    ) -> None:
        mocker.patch("polars.read_database_uri", side_effect=exception)

        with caplog.at_level(expected_log_level):
            result = retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]

        log_records = [record for record in caplog.records if record.levelname == expected_log_level]
        assert any(expected_log_msg in record.message for record in log_records)

    @pytest.mark.parametrize(
        "exception, expected_error_msg",
        [
            (
                RuntimeError("Polars-to-Pandas conversion failed"),
                "Error converting Polars DataFrame to Pandas DataFrame - Error N/A: Polars-to-Pandas conversion failed",
            ),
            (
                ValueError("Invalid Pandas data structure"),
                "Error converting Polars DataFrame to Pandas DataFrame - Error N/A: Invalid Pandas data structure",
            ),
        ],
    )
    def test_run_pandas_conversion_error(
        self,
        retriever: SnowflakeTableRetriever,
        caplog: pytest.LogCaptureFixture,
        mocker: Mock,
        exception: Exception,
        expected_error_msg: str,
        empty_response: Dict[str, Any],
        toy_polars_df: pl.DataFrame,
    ) -> None:
        mocker.patch.object(toy_polars_df, "to_pandas", side_effect=exception)
        mocker.patch("polars.read_database_uri", return_value=toy_polars_df)

        with caplog.at_level("ERROR"):
            result = retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]
        assert any(expected_error_msg in record.message for record in caplog.records)

    def test_run_happy_path(
        self,
        retriever: SnowflakeTableRetriever,
        mocker: Mock,
        toy_polars_df: pl.DataFrame,
        toy_pandas_df: pd.DataFrame,
        expected_markdown: str,
    ) -> None:
        mocker.patch("polars.read_database_uri", return_value=toy_polars_df)
        mocker.patch.object(toy_polars_df, "to_pandas", return_value=toy_pandas_df)
        mocker.patch.object(SnowflakeTableRetriever, "_polars_to_md", return_value=expected_markdown)

        result = retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == expected_markdown

    def test_empty_query(self, retriever: SnowflakeTableRetriever, empty_response: Dict[str, Any]) -> None:
        result = retriever.run(query="")
        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]

    def test_non_string_query(self, retriever: SnowflakeTableRetriever, empty_response: Dict[str, Any]) -> None:
        result = retriever.run(query=123)
        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]

    def test_empty_dataframe_result(
        self, retriever: SnowflakeTableRetriever, mocker: Mock, empty_response: Dict[str, Any]
    ) -> None:
        empty_df = pl.DataFrame()
        mocker.patch("polars.read_database_uri", return_value=empty_df)

        result = retriever.run(query="SELECT * FROM empty_table")

        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]

    def test_uri_construction_error(
        self, retriever: SnowflakeTableRetriever, mocker: Mock, empty_response: Dict[str, Any]
    ) -> None:
        mocker.patch.object(
            SnowflakeTableRetriever, "_snowflake_uri_constructor", side_effect=RuntimeError("Failed to construct URI")
        )

        result = retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(empty_response["dataframe"])
        assert result["table"] == empty_response["table"]

    def test_polars_to_md_error(
        self, retriever: SnowflakeTableRetriever, mocker: Mock, toy_polars_df: pl.DataFrame, toy_pandas_df: pd.DataFrame
    ) -> None:
        mocker.patch("polars.read_database_uri", return_value=toy_polars_df)
        mocker.patch.object(toy_polars_df, "to_pandas", return_value=toy_pandas_df)
        mocker.patch.object(
            SnowflakeTableRetriever, "_polars_to_md", side_effect=RuntimeError("Markdown conversion failed")
        )

        result = retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == ""

    def test_run_with_markdown_parameter(
        self, mocker: Mock, toy_polars_df: pl.DataFrame, toy_pandas_df: pd.DataFrame, expected_markdown: str
    ) -> None:
        mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})
        retriever = SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            database="test_db",
            return_markdown=False,
        )

        mocker.patch("polars.read_database_uri", return_value=toy_polars_df)
        mocker.patch.object(toy_polars_df, "to_pandas", return_value=toy_pandas_df)
        mocker.patch.object(SnowflakeTableRetriever, "_polars_to_md", return_value=expected_markdown)

        result = retriever.run(query="SELECT * FROM table_name", return_markdown=True)
        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == expected_markdown

        result = retriever.run(query="SELECT * FROM table_name")
        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == ""

        retriever.return_markdown = True
        result = retriever.run(query="SELECT * FROM table_name", return_markdown=False)
        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == ""

    def test_polars_to_md_empty_dataframe(self, retriever: SnowflakeTableRetriever) -> None:
        empty_df = pl.DataFrame()
        result = SnowflakeTableRetriever._polars_to_md(empty_df)
        assert result == ""

    def test_masked_uri_logging(
        self, retriever: SnowflakeTableRetriever, mocker: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger_mock = mocker.patch.object(
            logging.getLogger("haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever"), "info"
        )
        uri = retriever._snowflake_uri_constructor()

        assert uri is not None
        assert "test_api_key" in uri

        logger_mock.assert_any_call(
            "Constructed Snowflake URI: {masked_uri}", masked_uri=uri.replace("test_api_key", "***REDACTED***")
        )

    def test_custom_login_timeout(self, mocker: Mock) -> None:
        mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})
        custom_timeout = 120
        retriever = SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            database="test_db",
            login_timeout=custom_timeout,
        )

        uri = retriever._snowflake_uri_constructor()
        expected_uri = f"snowflake://test_user:test_api_key@test_account/test_db?login_timeout={custom_timeout}"
        assert uri == expected_uri
