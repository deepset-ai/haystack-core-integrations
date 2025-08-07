# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

import polars as pl
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from pandas import DataFrame

logger = logging.getLogger(__name__)


@component
class SnowflakeTableRetriever:
    """
    Connects to a Snowflake database to execute a SQL query using ADBC and Polars.
    Returns the results as a Pandas DataFrame (converted from a Polars DataFrame)
    along with a Markdown-formatted string.
    For more information, see [Polars documentation](https://docs.pola.rs/api/python/dev/reference/api/polars.read_database_uri.html).
    and [ADBC documentation](https://arrow.apache.org/adbc/main/driver/snowflake.html).

    ### Usage example:

    ```python
    executor = SnowflakeTableRetriever(
        user="<ACCOUNT-USER>",
        account="<ACCOUNT-IDENTIFIER>",
        api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
        database="<DATABASE-NAME>",
        db_schema="<SCHEMA-NAME>",
        warehouse="<WAREHOUSE-NAME>",
    )

    query = "SELECT * FROM table_name"
    results = executor.run(query=query)

    >> print(results["dataframe"].head(2))

        column1  column2        column3
    0     123   'data1'  2024-03-20
    1     456   'data2'  2024-03-21

    >> print(results["table"])

    shape: (3, 3)
    | column1 | column2 | column3    |
    |---------|---------|------------|
    | int     | str     | date       |
    |---------|---------|------------|
    | 123     | data1   | 2024-03-20 |
    | 456     | data2   | 2024-03-21 |
    | 789     | data3   | 2024-03-22 |
    ```
    """

    def __init__(
        self,
        user: str,
        account: str,
        api_key: Secret = Secret.from_env_var("SNOWFLAKE_API_KEY"),  # noqa: B008
        database: Optional[str] = None,
        db_schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        login_timeout: Optional[int] = 60,
        return_markdown: bool = True,
    ) -> None:
        """
        :param user: User's login.
        :param account: Snowflake account identifier.
        :param api_key: Snowflake account password.
        :param database: Name of the database to use.
        :param db_schema: Name of the schema to use.
        :param warehouse: Name of the warehouse to use.
        :param login_timeout: Timeout in seconds for login.
        :param return_markdown: Whether to return a Markdown-formatted string of the DataFrame.
        """

        self.user = user
        self.account = account
        self.api_key = api_key
        self.database = database
        self.db_schema = db_schema
        self.warehouse = warehouse
        self.login_timeout = login_timeout or 60
        self.return_markdown = return_markdown

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(  # type: ignore
            self,
            user=self.user,
            account=self.account,
            api_key=self.api_key.to_dict(),
            database=self.database,
            db_schema=self.db_schema,
            warehouse=self.warehouse,
            login_timeout=self.login_timeout,
            return_markdown=self.return_markdown,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnowflakeTableRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)

    def _snowflake_uri_constructor(self) -> str:
        """
        Constructs the Snowflake connection URI.

        Format: "snowflake://user:password@account/database/schema?warehouse=warehouse"

        :raises ValueError: If required credentials (`user` or `account`) are missing.
        :returns: A formatted Snowflake connection URI.
        """
        if not self.user or not self.account:
            msg = "Missing required Snowflake connection parameters: user and account."
            raise ValueError(msg)

        uri = f"snowflake://{self.user}:{self.api_key.resolve_value()}@{self.account}"
        if self.database:
            uri += f"/{self.database}"
            if self.db_schema:
                uri += f"/{self.db_schema}"
        uri += "?"
        if self.warehouse:
            uri += f"warehouse={self.warehouse}&"
        uri += f"login_timeout={self.login_timeout}&"
        uri = uri.rstrip("&?")

        # Logging placeholder for the actual password
        masked_uri = uri
        if resolved_api_key := self.api_key.resolve_value():
            masked_uri = uri.replace(resolved_api_key, "***REDACTED***")
        logger.info("Constructed Snowflake URI: {masked_uri}", masked_uri=masked_uri)
        return uri

    @staticmethod
    def _polars_to_md(data: pl.DataFrame) -> str:
        """
        Converts a Polars DataFrame to a Markdown-formatted string.
        Uses Polars' built-in table formatting for efficient conversion.

        :param data: The Polars DataFrame to convert.
        :returns: A Markdown-formatted string if `data` is not empty; otherwise, an empty string.
        """
        if data.is_empty():
            return ""  # No markdown for empty DataFrame.

        try:
            with pl.Config(tbl_formatting="MARKDOWN"):
                return str(data)
        except Exception as e:
            logger.error(
                "Error converting Polars DataFrame to Markdown - Error {errno}: {error_msg}",
                errno=getattr(e, "errno", "N/A"),
                error_msg=getattr(e, "msg", str(e)),
                exc_info=True,
            )
            return ""

    @staticmethod
    def _empty_response() -> Dict[str, Any]:
        """Returns a standardized empty response.

        :returns:
            A dictionary with the following keys:
            - `dataframe`: An empty Pandas DataFrame.
            - `table`: An empty Markdown string.
        """
        return {"dataframe": DataFrame(), "table": ""}

    @component.output_types(dataframe=DataFrame, table=str)
    def run(self, query: str, return_markdown: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executes a SQL query against a Snowflake database using ADBC and Polars.

        :param query: The SQL query to execute.
        :param return_markdown: Whether to return a Markdown-formatted string of the DataFrame.
            If not provided, uses the value set during initialization.
        :returns: A dictionary containing:
            - `"dataframe"`: A Pandas DataFrame with the query results.
            - `"table"`: A Markdown-formatted string representation of the DataFrame.
        """
        # Validate SQL query
        if not query:
            logger.warning("Empty query provided, returning empty DataFrame")
            return self._empty_response()

        if not isinstance(query, str):
            logger.warning("Query is not a string, returning empty DataFrame")
            return self._empty_response()

        logger.info("Starting query execution")
        logger.info("Query: {query}", query=query)

        try:
            # Construct the URI using the helper method
            uri = self._snowflake_uri_constructor()
        except Exception as e:
            logger.error(
                "Error constructing Snowflake URI - Error {errno}: {error_msg}",
                errno=getattr(e, "errno", "N/A"),
                error_msg=getattr(e, "msg", str(e)),
                exc_info=True,
            )
            return self._empty_response()

        try:
            # Execute the query via Polars using the ADBC engine
            data = pl.read_database_uri(query, uri, engine="adbc")

            # Check for valid data and schema before proceeding
            if data.is_empty() or data.schema is None:
                logger.warning("Query returned an empty DataFrame or invalid schema")
                return self._empty_response()

            logger.info(
                "Query execution completed. Polars DataFrame shape: {shape}, columns: {columns}",
                shape=data.shape,
                columns=data.columns,
            )
        except Exception as e:
            error_msg = getattr(e, "msg", str(e))  # Get error message

            # Check if the error message indicates a SQL compilation issue
            if "SQL compilation error" in error_msg or "invalid identifier" in error_msg:
                logger.warning(
                    "SQL compilation error encountered: {error_msg}",
                    error_msg=error_msg,
                    exc_info=False,  # Avoid full traceback in logs for expected warnings
                )
            else:
                logger.error(
                    "Error executing query via ADBC - Error {errno}: {error_msg}",
                    errno=getattr(e, "errno", "N/A"),
                    error_msg=error_msg,
                    exc_info=True,  # Preserve traceback for debugging
                )

            return self._empty_response()

        # Convert Polars DataFrame to Pandas DataFrame deliberately for downstream compatibility
        try:
            pandas_df = data.to_pandas()
            column_info = f", columns: {pandas_df.columns.tolist()}" if pandas_df.shape[1] > 0 else ""
            logger.info(
                "Converted to Pandas DataFrame. Shape: {shape}{columns}",
                shape=pandas_df.shape,
                columns=column_info,
            )
        except Exception as e:
            logger.error(
                "Error converting Polars DataFrame to Pandas DataFrame - Error {errno}: {error_msg}",
                errno=getattr(e, "errno", "N/A"),
                error_msg=getattr(e, "msg", str(e)),
                exc_info=True,
            )
            return self._empty_response()

        # Convert Polars DataFrame to Markdown **only if return_markdown is True**
        markdown_str = ""

        should_return_markdown = self.return_markdown if return_markdown is None else return_markdown
        if should_return_markdown:
            try:
                markdown_str = self._polars_to_md(data)
            except Exception as e:
                logger.error(
                    "Error converting Polars DataFrame to Markdown - Error {errno}: {error_msg}",
                    errno=getattr(e, "errno", "N/A"),
                    error_msg=getattr(e, "msg", str(e)),
                    exc_info=True,
                )

        return {"dataframe": pandas_df, "table": markdown_str}
