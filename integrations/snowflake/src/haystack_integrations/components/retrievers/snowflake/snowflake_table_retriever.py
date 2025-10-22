# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Literal, Optional
from urllib.parse import quote_plus

import polars as pl
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from pandas import DataFrame

import snowflake

from .auth import SnowflakeAuthenticator

logger = logging.getLogger(__name__)


@component
class SnowflakeTableRetriever:
    """
    Connects to a Snowflake database to execute a SQL query using ADBC and Polars.
    Returns the results as a Pandas DataFrame (converted from a Polars DataFrame)
    along with a Markdown-formatted string.
    For more information, see [Polars documentation](https://docs.pola.rs/api/python/dev/reference/api/polars.read_database_uri.html).
    and [ADBC documentation](https://arrow.apache.org/adbc/main/driver/snowflake.html).

    ### Usage examples:

    #### Password Authentication:
    ```python
    executor = SnowflakeTableRetriever(
        user="<ACCOUNT-USER>",
        account="<ACCOUNT-IDENTIFIER>",
        authenticator="SNOWFLAKE",
        api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
        database="<DATABASE-NAME>",
        db_schema="<SCHEMA-NAME>",
        warehouse="<WAREHOUSE-NAME>",
    )
    executor.warm_up()
    ```

    #### Key-pair Authentication (MFA):
    ```python
    executor = SnowflakeTableRetriever(
        user="<ACCOUNT-USER>",
        account="<ACCOUNT-IDENTIFIER>",
        authenticator="SNOWFLAKE_JWT",
        private_key_file=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE"),
        private_key_file_pwd=Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD"),
        database="<DATABASE-NAME>",
        db_schema="<SCHEMA-NAME>",
        warehouse="<WAREHOUSE-NAME>",
    )
    executor.warm_up()
    ```

    #### OAuth Authentication (MFA):
    ```python
    executor = SnowflakeTableRetriever(
        user="<ACCOUNT-USER>",
        account="<ACCOUNT-IDENTIFIER>",
        authenticator="OAUTH",
        oauth_client_id=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID"),
        oauth_client_secret=Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET"),
        oauth_token_request_url="<TOKEN-REQUEST-URL>",
        database="<DATABASE-NAME>",
        db_schema="<SCHEMA-NAME>",
        warehouse="<WAREHOUSE-NAME>",
    )
    executor.warm_up()
    ```

    #### Running queries:
    ```python
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
        authenticator: Literal["SNOWFLAKE", "SNOWFLAKE_JWT", "OAUTH"] = "SNOWFLAKE",
        api_key: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_API_KEY", strict=False),  # noqa: B008
        database: Optional[str] = None,
        db_schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        login_timeout: Optional[int] = 60,
        return_markdown: bool = True,
        private_key_file: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_FILE", strict=False),  # noqa: B008
        private_key_file_pwd: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_PRIVATE_KEY_PWD", strict=False),  # noqa: B008
        oauth_client_id: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_ID", strict=False),  # noqa: B008
        oauth_client_secret: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_OAUTH_CLIENT_SECRET", strict=False),  # noqa: B008
        oauth_token_request_url: Optional[str] = None,
        oauth_authorization_url: Optional[str] = None,
    ) -> None:
        """
        :param user: User's login.
        :param account: Snowflake account identifier.
        :param authenticator: Authentication method. Required. Options: "SNOWFLAKE" (password),
            "SNOWFLAKE_JWT" (key-pair), or "OAUTH".
        :param api_key: Snowflake account password. Required for SNOWFLAKE authentication.
        :param database: Name of the database to use.
        :param db_schema: Name of the schema to use.
        :param warehouse: Name of the warehouse to use.
        :param login_timeout: Timeout in seconds for login.
        :param return_markdown: Whether to return a Markdown-formatted string of the DataFrame.
        :param private_key_file: Secret containing the path to private key file.
            Required for SNOWFLAKE_JWT authentication.
        :param private_key_file_pwd: Secret containing the passphrase for private key file.
            Required only when the private key file is encrypted.
        :param oauth_client_id: Secret containing the OAuth client ID.
            Required for OAUTH authentication.
        :param oauth_client_secret: Secret containing the OAuth client secret.
            Required for OAUTH authentication.
        :param oauth_token_request_url: OAuth token request URL for Client Credentials flow.
        :param oauth_authorization_url: OAuth authorization URL for Authorization Code flow.
        """

        self.user = user
        self.account = account
        self.api_key = api_key
        self.database = database
        self.db_schema = db_schema
        self.warehouse = warehouse
        self.login_timeout = login_timeout or 60
        self.return_markdown = return_markdown
        self.authenticator = authenticator
        self.private_key_file = private_key_file
        self.private_key_file_pwd = private_key_file_pwd
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self.oauth_token_request_url = oauth_token_request_url
        self.oauth_authorization_url = oauth_authorization_url
        self.authenticator_handler: Optional[SnowflakeAuthenticator] = None
        self._warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the component by initializing the authenticator handler and testing the database connection.
        """
        if self._warmed_up:
            return
        self.authenticator_handler = SnowflakeAuthenticator(
            authenticator=self.authenticator,
            api_key=self.api_key,
            private_key_file=self.private_key_file,
            private_key_file_pwd=self.private_key_file_pwd,
            oauth_client_id=self.oauth_client_id,
            oauth_client_secret=self.oauth_client_secret,
            oauth_token_request_url=self.oauth_token_request_url,
            oauth_authorization_url=self.oauth_authorization_url,
        )

        # Test connection during initialization to verify credentials
        if not self.authenticator_handler.test_connection(user=self.user, account=self.account, database=self.database):
            msg = "Failed to connect to Snowflake with provided credentials"
            raise ConnectionError(msg)

        self._warmed_up = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data: Dict[str, Any] = {
            "user": self.user,
            "account": self.account,
            "database": self.database,
            "db_schema": self.db_schema,
            "warehouse": self.warehouse,
            "login_timeout": self.login_timeout,
            "return_markdown": self.return_markdown,
            "authenticator": self.authenticator,
            "oauth_token_request_url": self.oauth_token_request_url,
            "oauth_authorization_url": self.oauth_authorization_url,
            "api_key": self.api_key.to_dict() if self.api_key else None,
            "private_key_file": self.private_key_file.to_dict() if self.private_key_file else None,
            "private_key_file_pwd": self.private_key_file_pwd.to_dict() if self.private_key_file_pwd else None,
            "oauth_client_id": self.oauth_client_id.to_dict() if self.oauth_client_id else None,
            "oauth_client_secret": self.oauth_client_secret.to_dict() if self.oauth_client_secret else None,
        }
        return default_to_dict(self, **data)

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
        secret_fields = [
            "api_key",
            "private_key_file",
            "private_key_file_pwd",
            "oauth_client_id",
            "oauth_client_secret",
        ]
        deserialize_secrets_inplace(init_params, secret_fields)
        return default_from_dict(cls, data)

    def _snowflake_uri_constructor(self) -> str:
        """
        Constructs the Snowflake connection URI based on the authentication method.

        Formats:
        - Password: "snowflake://user:password@account/database/schema?warehouse=warehouse"
        - Key-pair JWT: "snowflake://user@account/database/schema?warehouse=warehouse&authenticator=SNOWFLAKE_JWT&private_key_file=path&private_key_file_pwd=pwd"
        - OAuth: "snowflake://user@account/database/schema?warehouse=warehouse&authenticator=OAUTH&oauth_client_id=id&oauth_client_secret=secret"

        :raises ValueError: If required credentials are missing.
        :returns: A formatted Snowflake connection URI.
        """
        if not self.user or not self.account:
            msg = "Missing required Snowflake connection parameters: user and account."
            raise ValueError(msg)

        # Base URI construction - encode user and account for URI safety
        encoded_user = quote_plus(self.user)
        encoded_account = quote_plus(self.account)

        # We ignore the mypy error since it doesn't know that self.authenticator_handler has been set at this point
        password = self.authenticator_handler.get_password_for_uri()  # type: ignore[union-attr]
        if password:
            # Traditional password authentication - encode password
            encoded_password = quote_plus(password)
            uri = f"snowflake://{encoded_user}:{encoded_password}@{encoded_account}"
        elif self.authenticator == "SNOWFLAKE_JWT":
            # For JWT with ADBC, use account-only URI and pass username as parameter
            # This avoids ADBC interpreting user@account as empty password auth
            uri = f"snowflake://{encoded_account}"
        else:
            # Other MFA authentication methods (OAuth, etc.)
            uri = f"snowflake://{encoded_user}@{encoded_account}"

        # Add database and schema
        if self.database:
            uri += f"/{self.database}"
            if self.db_schema:
                uri += f"/{self.db_schema}"

        # Add query parameters
        params = []
        if self.warehouse:
            params.append(f"warehouse={self.warehouse}")
        params.append(f"login_timeout={self.login_timeout}")

        # Add authentication-specific parameters (pass user for JWT ADBC support)
        # We ignore the mypy error since it doesn't know that self.authenticator_handler has been set at this point
        auth_params = self.authenticator_handler.build_auth_params(user=self.user)  # type: ignore[union-attr]
        params.extend(auth_params)

        if params:
            uri += "?" + "&".join(params)

        # Create masked URI for logging
        masked_uri = self._create_masked_uri(uri)
        logger.info("Constructed Snowflake URI: {masked_uri}", masked_uri=masked_uri)
        return uri

    def _create_masked_uri(self, uri: str) -> str:
        """
        Creates a masked version of the URI for safe logging.

        :param uri: Original URI.
        :returns: URI with sensitive information masked.
        """
        masked_uri = uri

        # Mask password if present
        if self.authenticator == "SNOWFLAKE":
            # We ignore the mypy error since it doesn't know that self.authenticator_handler has been set at this point
            password = self.authenticator_handler.get_password_for_uri()  # type: ignore[union-attr]
            if password:
                encoded_password = quote_plus(password)
                masked_uri = masked_uri.replace(encoded_password, "***REDACTED***")

        # Mask authentication secrets in parameters
        if "?" in masked_uri:
            base_uri, query_params = masked_uri.split("?", 1)
            param_list = query_params.split("&")
            masked_params = self.authenticator_handler.create_masked_params(param_list)  # type: ignore[union-attr]
            masked_uri = base_uri + "?" + "&".join(masked_params)

        return masked_uri

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
                exc_info=False,
            )
            return ""

    def _execute_query_with_connector(self, query: str) -> Optional[pl.DataFrame]:
        """
        Executes a query using snowflake-connector-python directly (for JWT authentication).
        This bypasses ADBC compatibility issues.

        :param query: SQL query to execute.
        :returns: Polars DataFrame with results, or None if execution fails.
        """
        try:
            # Build connection parameters
            conn_params: Dict[str, Any] = {
                "user": self.user,
                "account": self.account,
                "authenticator": self.authenticator.lower(),
            }

            if self.database:
                conn_params["database"] = self.database
            if self.db_schema:
                conn_params["schema"] = self.db_schema
            if self.warehouse:
                conn_params["warehouse"] = self.warehouse

            # Add JWT-specific parameters
            if self.authenticator == "SNOWFLAKE_JWT":
                # We ignore the mypy error since it doesn't know that self.authenticator_handler has been set at this
                # point
                if self.authenticator_handler.private_key_file:  # type: ignore[union-attr]
                    conn_params["private_key_file"] = self.authenticator_handler.private_key_file  # type: ignore[union-attr]

                # We ignore the mypy error since it doesn't know that self.authenticator_handler has been set at this
                # point
                if self.authenticator_handler.private_key_file_pwd:  # type: ignore[union-attr]
                    conn_params["private_key_file_pwd"] = self.authenticator_handler.private_key_file_pwd  # type: ignore[union-attr]

            # Connect and execute query
            conn = snowflake.connector.connect(**conn_params)
            try:
                cursor = conn.cursor()
                cursor.execute(query)

                # Fetch results and convert to Polars DataFrame
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()

                # Convert to Polars DataFrame
                if data:
                    data_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
                    return pl.DataFrame(data_dict)
                else:
                    return pl.DataFrame()
            finally:
                conn.close()

        except Exception as e:
            logger.error(
                "Error executing query with snowflake-connector - Error {errno}: {error_msg}",
                errno=getattr(e, "errno", "N/A"),
                error_msg=str(e),
                exc_info=False,  # Avoid displaying the whole traceback to the user
            )
            return None

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
        if not self._warmed_up:
            msg = "SnowflakeTableRetriever not warmed up. Please call `warm_up()` before running queries."
            raise RuntimeError(msg)

        # Validate SQL query
        if not query:
            logger.warning("Empty query provided, returning empty DataFrame")
            return self._empty_response()

        if not isinstance(query, str):
            logger.warning("Query is not a string, returning empty DataFrame")
            return self._empty_response()

        logger.info("Starting query execution")
        logger.info("Query: {query}", query=query)

        # Use snowflake-connector-python directly for JWT to bypass ADBC compatibility issues
        if self.authenticator == "SNOWFLAKE_JWT":
            logger.info("Using snowflake-connector-python for JWT authentication")
            data = self._execute_query_with_connector(query)

            if data is None:
                logger.error("Query execution failed with snowflake-connector")
                return self._empty_response()
        else:
            # Use ADBC via Polars for other authentication methods
            try:
                # Construct the URI using the helper method
                uri = self._snowflake_uri_constructor()
            except Exception as e:
                logger.error(
                    "Error constructing Snowflake URI - Error {errno}: {error_msg}",
                    errno=getattr(e, "errno", "N/A"),
                    error_msg=getattr(e, "msg", str(e)),
                    exc_info=False,
                )
                return self._empty_response()

            try:
                # Execute the query via Polars using the ADBC engine
                data = pl.read_database_uri(query, uri, engine="adbc")
            except Exception as e:
                error_msg = getattr(e, "msg", str(e))

                # Check if the error message indicates a SQL compilation issue
                if "SQL compilation error" in error_msg or "invalid identifier" in error_msg:
                    logger.warning(
                        "SQL compilation error encountered: {error_msg}",
                        error_msg=error_msg,
                        exc_info=False,
                    )
                else:
                    logger.error(
                        "Error executing query via ADBC - Error {errno}: {error_msg}",
                        errno=getattr(e, "errno", "N/A"),
                        error_msg=error_msg,
                        exc_info=False,
                    )

                return self._empty_response()

        # Check for valid data and schema before proceeding
        if data.is_empty() or data.schema is None:
            logger.warning("Query returned an empty DataFrame or invalid schema")
            return self._empty_response()

        logger.info(
            "Query execution completed. Polars DataFrame shape: {shape}, columns: {columns}",
            shape=data.shape,
            columns=data.columns,
        )

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
                exc_info=False,
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
                    exc_info=False,
                )

        return {"dataframe": pandas_df, "table": markdown_str}
