import os
from pathlib import Path
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
    # Mock the connection test to avoid requiring actual Snowflake connection during tests
    mocker.patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
        return_value=True,
    )
    return SnowflakeTableRetriever(
        user="test_user",
        account="test_account",
        authenticator="SNOWFLAKE",
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


@pytest.fixture
def jwt_retriever(mocker: Mock) -> SnowflakeTableRetriever:
    mocker.patch.dict(
        os.environ, {"SNOWFLAKE_PRIVATE_KEY_FILE": "/path/to/key.pem", "SNOWFLAKE_PRIVATE_KEY_PWD": "test_password"}
    )
    # Mock the connection test to avoid requiring actual Snowflake connection during tests
    mocker.patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
        return_value=True,
    )
    return SnowflakeTableRetriever(
        user="test_user",
        account="test_account",
        authenticator="SNOWFLAKE_JWT",
        private_key_file=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE"),
        private_key_file_pwd=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD"),
        database="test_db",
        db_schema="test_schema",
        warehouse="test_warehouse",
        return_markdown=True,
    )


@pytest.fixture
def oauth_retriever(mocker: Mock) -> SnowflakeTableRetriever:
    mocker.patch.dict(
        os.environ,
        {"SNOWFLAKE_OAUTH_CLIENT_ID": "test_client_id", "SNOWFLAKE_OAUTH_CLIENT_SECRET": "test_client_secret"},
    )
    # Mock the connection test to avoid requiring actual Snowflake connection during tests
    mocker.patch(
        "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
        return_value=True,
    )
    return SnowflakeTableRetriever(
        user="test_user",
        account="test_account",
        authenticator="OAUTH",
        oauth_client_id=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID"),
        oauth_client_secret=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET"),
        oauth_token_request_url="https://test.snowflakecomputing.com/oauth/token-request",
        database="test_db",
        db_schema="test_schema",
        warehouse="test_warehouse",
        return_markdown=True,
    )


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
        # Mock connection test for direct instantiation
        mocker.patch(
            "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
            return_value=True,
        )

        retriever = SnowflakeTableRetriever(
            user=user,
            account=account,
            authenticator="SNOWFLAKE",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            database=db_name,
            db_schema=schema_name,
            warehouse=warehouse_name,
        )

        if should_raise:
            with pytest.raises(
                ValueError,
                match=r"Missing required Snowflake connection parameters: user and account\.",
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
        mocker.patch(
            "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
            return_value=True,
        )
        retriever = SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            authenticator="SNOWFLAKE",
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
        mocker.patch(
            "haystack_integrations.components.retrievers.snowflake.snowflake_table_retriever.SnowflakeTableRetriever.test_connection",
            return_value=True,
        )
        custom_timeout = 120
        retriever = SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            authenticator="SNOWFLAKE",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            database="test_db",
            login_timeout=custom_timeout,
        )

        uri = retriever._snowflake_uri_constructor()
        expected_uri = f"snowflake://test_user:test_api_key@test_account/test_db?login_timeout={custom_timeout}"
        assert uri == expected_uri

    def test_jwt_authentication_serialization(self, jwt_retriever: SnowflakeTableRetriever) -> None:
        serialized = jwt_retriever.to_dict()
        init_params = serialized["init_parameters"]

        assert init_params["authenticator"] == "SNOWFLAKE_JWT"
        assert "private_key_file" in init_params
        assert "private_key_file_pwd" in init_params

        deserialized = SnowflakeTableRetriever.from_dict(serialized)
        assert isinstance(deserialized, SnowflakeTableRetriever)
        assert deserialized.authenticator == "SNOWFLAKE_JWT"

    def test_oauth_authentication_serialization(self, oauth_retriever: SnowflakeTableRetriever) -> None:
        serialized = oauth_retriever.to_dict()
        init_params = serialized["init_parameters"]

        assert init_params["authenticator"] == "OAUTH"
        assert "oauth_client_id" in init_params
        assert "oauth_client_secret" in init_params
        assert init_params["oauth_token_request_url"] == "https://test.snowflakecomputing.com/oauth/token-request"

        deserialized = SnowflakeTableRetriever.from_dict(serialized)
        assert isinstance(deserialized, SnowflakeTableRetriever)
        assert deserialized.authenticator == "OAUTH"

    def test_jwt_uri_construction(self, jwt_retriever: SnowflakeTableRetriever) -> None:
        uri = jwt_retriever._snowflake_uri_constructor()
        # JWT uses account-only URI with username and auth params as query parameters
        assert uri.startswith("snowflake://test_account/test_db/test_schema?")
        assert "warehouse=test_warehouse" in uri
        assert "login_timeout=60" in uri
        assert "adbc.snowflake.sql.auth_type=auth_jwt" in uri
        assert "username=test_user" in uri

    def test_oauth_uri_construction(self, oauth_retriever: SnowflakeTableRetriever) -> None:
        uri = oauth_retriever._snowflake_uri_constructor()
        expected_uri = "snowflake://test_user@test_account/test_db/test_schema?warehouse=test_warehouse&login_timeout=60&authenticator=OAUTH&oauth_client_id=test_client_id&oauth_client_secret=test_client_secret&oauth_token_request_url=https://test.snowflakecomputing.com/oauth/token-request"
        assert uri == expected_uri

    def test_masked_uri_logging_jwt(self, jwt_retriever: SnowflakeTableRetriever) -> None:
        uri = jwt_retriever._snowflake_uri_constructor()
        masked_uri = jwt_retriever._create_masked_uri(uri)

        assert "test_password" not in masked_uri
        assert "***REDACTED***" in masked_uri

    def test_masked_uri_logging_oauth(self, oauth_retriever: SnowflakeTableRetriever) -> None:
        uri = oauth_retriever._snowflake_uri_constructor()
        masked_uri = oauth_retriever._create_masked_uri(uri)

        assert "test_client_secret" not in masked_uri
        assert "***REDACTED***" in masked_uri

    @pytest.mark.parametrize(
        "authenticator, missing_param, expected_error",
        [
            ("SNOWFLAKE_JWT", "private_key_file", "private_key_file is required for SNOWFLAKE_JWT authentication"),
            ("OAUTH", "oauth_client_id", "oauth_client_id is required for OAUTH authentication"),
            ("OAUTH", "oauth_client_secret", "oauth_client_secret is required for OAUTH authentication"),
            ("SNOWFLAKE", "api_key", "api_key is required for SNOWFLAKE \\(password\\) authentication"),
        ],
    )
    def test_authentication_validation_errors(
        self, mocker: Mock, authenticator: str, missing_param: str, expected_error: str
    ) -> None:
        # Set up environment variables, excluding the one being tested as missing
        env_vars = {
            "SNOWFLAKE_PRIVATE_KEY_FILE": "/path/to/key.pem",
            "SNOWFLAKE_PRIVATE_KEY_PWD": "test_password",
            "SNOWFLAKE_OAUTH_CLIENT_ID": "test_client_id",
            "SNOWFLAKE_OAUTH_CLIENT_SECRET": "test_client_secret",
        }

        # Only set SNOWFLAKE_API_KEY if we're not testing its absence
        if not (authenticator == "SNOWFLAKE" and missing_param == "api_key"):
            env_vars["SNOWFLAKE_API_KEY"] = "test_api_key"

        mocker.patch.dict(os.environ, env_vars, clear=True)

        kwargs = {
            "user": "test_user",
            "account": "test_account",
            "authenticator": authenticator,
        }

        if authenticator == "SNOWFLAKE_JWT":
            if missing_param != "private_key_file":
                kwargs["private_key_file"] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE")
            if missing_param != "private_key_file_pwd":
                kwargs["private_key_file_pwd"] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD")
        elif authenticator == "OAUTH":
            if missing_param != "oauth_client_id":
                kwargs["oauth_client_id"] = Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID")
            if missing_param != "oauth_client_secret":
                kwargs["oauth_client_secret"] = Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET")
        elif authenticator == "SNOWFLAKE":
            if missing_param != "api_key":
                kwargs["api_key"] = Secret.from_env_var("SNOWFLAKE_API_KEY")

        with pytest.raises(ValueError, match=expected_error):
            SnowflakeTableRetriever(**kwargs)

    def test_jwt_authentication_happy_path(
        self,
        jwt_retriever: SnowflakeTableRetriever,
        mocker: Mock,
        toy_polars_df: pl.DataFrame,
        toy_pandas_df: pd.DataFrame,
        expected_markdown: str,
    ) -> None:
        # JWT uses _execute_query_with_connector instead of ADBC
        mocker.patch.object(SnowflakeTableRetriever, "_execute_query_with_connector", return_value=toy_polars_df)
        mocker.patch.object(toy_polars_df, "to_pandas", return_value=toy_pandas_df)
        mocker.patch.object(SnowflakeTableRetriever, "_polars_to_md", return_value=expected_markdown)

        result = jwt_retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == expected_markdown

    def test_oauth_authentication_happy_path(
        self,
        oauth_retriever: SnowflakeTableRetriever,
        mocker: Mock,
        toy_polars_df: pl.DataFrame,
        toy_pandas_df: pd.DataFrame,
        expected_markdown: str,
    ) -> None:
        mocker.patch("polars.read_database_uri", return_value=toy_polars_df)
        mocker.patch.object(toy_polars_df, "to_pandas", return_value=toy_pandas_df)
        mocker.patch.object(SnowflakeTableRetriever, "_polars_to_md", return_value=expected_markdown)

        result = oauth_retriever.run(query="SELECT * FROM table_name")

        assert result["dataframe"].equals(toy_pandas_df)
        assert result["table"] == expected_markdown

    def test_connection_success(self, mocker: Mock) -> None:
        # Create retriever without mocking test_connection
        mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})

        # Mock the actual snowflake.connector.connect function
        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        # Create retriever (test_connection will be called during init and will use the mock)
        SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            authenticator="SNOWFLAKE",
            api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
        )

        # Verify the connection was tested during initialization
        assert mock_connect.call_count >= 1
        mock_connection.close.assert_called()

    def test_connection_failure(self, mocker: Mock) -> None:
        # Mock the snowflake module import
        mocker.patch.dict(os.environ, {"SNOWFLAKE_API_KEY": "test_api_key"})

        mock_snowflake = mocker.Mock()
        mock_snowflake.connector.connect.side_effect = Exception("Connection failed")
        mocker.patch.dict("sys.modules", {"snowflake": mock_snowflake, "snowflake.connector": mock_snowflake.connector})

        # Should raise ConnectionError during initialization
        with pytest.raises(ConnectionError, match="Failed to connect to Snowflake"):
            SnowflakeTableRetriever(
                user="test_user",
                account="test_account",
                authenticator="SNOWFLAKE",
                api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
            )

    def test_connection_jwt_auth(self, mocker: Mock, tmp_path: Path) -> None:
        # Create a temporary key file
        key_file = tmp_path / "key.pem"
        key_file.write_text("-----BEGIN PRIVATE KEY-----\ntest_key_content\n-----END PRIVATE KEY-----")

        # Mock the snowflake module import and environment
        mocker.patch.dict(
            os.environ, {"SNOWFLAKE_PRIVATE_KEY_FILE": str(key_file), "SNOWFLAKE_PRIVATE_KEY_PWD": "test_password"}
        )

        # Mock the actual snowflake.connector.connect function
        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        # Create JWT retriever (test_connection will be called during init)
        SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            authenticator="SNOWFLAKE_JWT",
            private_key_file=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE"),
            private_key_file_pwd=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD"),
        )

        # Verify that JWT-specific parameters were passed
        assert mock_connect.call_count >= 1
        call_args = mock_connect.call_args[1]
        assert call_args["authenticator"] == "snowflake_jwt"
        assert "private_key_file" in call_args

    def test_connection_oauth_auth(self, mocker: Mock) -> None:
        # Mock the snowflake module import and environment
        mocker.patch.dict(
            os.environ,
            {"SNOWFLAKE_OAUTH_CLIENT_ID": "test_client_id", "SNOWFLAKE_OAUTH_CLIENT_SECRET": "test_client_secret"},
        )

        # Mock the actual snowflake.connector.connect function
        mock_connection = mocker.Mock()
        mock_connect = mocker.patch("snowflake.connector.connect", return_value=mock_connection)

        # Create OAuth retriever (test_connection will be called during init)
        SnowflakeTableRetriever(
            user="test_user",
            account="test_account",
            authenticator="OAUTH",
            oauth_client_id=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID"),
            oauth_client_secret=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET"),
        )

        # Verify that OAuth-specific parameters were passed
        assert mock_connect.call_count >= 1
        call_args = mock_connect.call_args[1]
        assert call_args["authenticator"] == "oauth"
        assert "oauth_client_id" in call_args
        assert "oauth_client_secret" in call_args
