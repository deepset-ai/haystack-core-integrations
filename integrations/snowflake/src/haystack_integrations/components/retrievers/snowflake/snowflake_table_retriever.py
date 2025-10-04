# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Literal, Optional, Union

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

    ### Usage examples:

    #### Password Authentication (default):
    ```python
    executor = SnowflakeTableRetriever(
        user="<ACCOUNT-USER>",
        account="<ACCOUNT-IDENTIFIER>",
        api_key=Secret.from_env_var("SNOWFLAKE_API_KEY"),
        database="<DATABASE-NAME>",
        db_schema="<SCHEMA-NAME>",
        warehouse="<WAREHOUSE-NAME>",
    )
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
        api_key: Optional[Secret] = Secret.from_env_var("SNOWFLAKE_API_KEY"),  # noqa: B008
        database: Optional[str] = None,
        db_schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        login_timeout: Optional[int] = 60,
        return_markdown: bool = True,
        authenticator: Optional[Literal["SNOWFLAKE", "SNOWFLAKE_JWT", "OAUTH"]] = None,
        private_key_file: Optional[Union[str, Secret]] = None,
        private_key_file_pwd: Optional[Union[str, Secret]] = None,
        oauth_client_id: Optional[Union[str, Secret]] = None,
        oauth_client_secret: Optional[Union[str, Secret]] = None,
        oauth_token_request_url: Optional[str] = None,
        oauth_authorization_url: Optional[str] = None,
    ) -> None:
        """
        :param user: User's login.
        :param account: Snowflake account identifier.
        :param api_key: Snowflake account password. Required for default password authentication.
        :param database: Name of the database to use.
        :param db_schema: Name of the schema to use.
        :param warehouse: Name of the warehouse to use.
        :param login_timeout: Timeout in seconds for login.
        :param return_markdown: Whether to return a Markdown-formatted string of the DataFrame.
        :param authenticator: Authentication method. Options: "SNOWFLAKE" (default password),
            "SNOWFLAKE_JWT" (key-pair), or "OAUTH".
        :param private_key_file: Path to private key file or Secret containing the path.
            Required for SNOWFLAKE_JWT authentication.
        :param private_key_file_pwd: Passphrase for private key file or Secret containing the passphrase.
            Required for SNOWFLAKE_JWT authentication.
        :param oauth_client_id: OAuth client ID or Secret containing the client ID.
            Required for OAUTH authentication.
        :param oauth_client_secret: OAuth client secret or Secret containing the client secret.
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

        # Authentication parameters
        self.authenticator = authenticator or "SNOWFLAKE"
        self.private_key_file = private_key_file
        self.private_key_file_pwd = private_key_file_pwd
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self.oauth_token_request_url = oauth_token_request_url
        self.oauth_authorization_url = oauth_authorization_url

        # Validate authentication parameters
        self._validate_auth_params()

    def _validate_auth_params(self) -> None:
        """
        Validates authentication parameters based on the chosen authentication method.

        :raises ValueError: If required parameters are missing for the selected authentication method.
        """
        if self.authenticator == "SNOWFLAKE_JWT":
            if not self.private_key_file:
                msg = "private_key_file is required for SNOWFLAKE_JWT authentication"
                raise ValueError(msg)
            if not self.private_key_file_pwd:
                msg = "private_key_file_pwd is required for SNOWFLAKE_JWT authentication"
                raise ValueError(msg)
        elif self.authenticator == "OAUTH":
            if not self.oauth_client_id:
                msg = "oauth_client_id is required for OAUTH authentication"
                raise ValueError(msg)
            if not self.oauth_client_secret:
                msg = "oauth_client_secret is required for OAUTH authentication"
                raise ValueError(msg)
        elif self.authenticator == "SNOWFLAKE":
            if not self.api_key:
                msg = "api_key is required for SNOWFLAKE (password) authentication"
                raise ValueError(msg)
            try:
                api_key_value = self.api_key.resolve_value()
                if not api_key_value:
                    msg = "api_key is required for SNOWFLAKE (password) authentication"
                    raise ValueError(msg)
            except ValueError as e:
                if "authentication environment variables are set" in str(e):
                    msg = "api_key is required for SNOWFLAKE (password) authentication"
                    raise ValueError(msg) from e
                raise

    def _resolve_secret_value(self, value: Optional[Union[str, Secret]]) -> Optional[str]:
        """
        Resolves a Secret value or returns the string value.

        :param value: String or Secret to resolve.
        :returns: Resolved string value or None.
        """
        if value is None:
            return None
        if isinstance(value, Secret):
            return value.resolve_value()
        return value

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
        }

        # Handle Secret fields
        if self.api_key:
            data["api_key"] = self.api_key.to_dict()
        if self.private_key_file:
            data["private_key_file"] = (
                self.private_key_file.to_dict() if isinstance(self.private_key_file, Secret) else self.private_key_file
            )
        if self.private_key_file_pwd:
            data["private_key_file_pwd"] = (
                self.private_key_file_pwd.to_dict()
                if isinstance(self.private_key_file_pwd, Secret)
                else self.private_key_file_pwd
            )
        if self.oauth_client_id:
            data["oauth_client_id"] = (
                self.oauth_client_id.to_dict() if isinstance(self.oauth_client_id, Secret) else self.oauth_client_id
            )
        if self.oauth_client_secret:
            data["oauth_client_secret"] = (
                self.oauth_client_secret.to_dict()
                if isinstance(self.oauth_client_secret, Secret)
                else self.oauth_client_secret
            )

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

        # Base URI construction
        if self.authenticator == "SNOWFLAKE" and self.api_key:
            # Traditional password authentication
            uri = f"snowflake://{self.user}:{self.api_key.resolve_value()}@{self.account}"
        else:
            # MFA authentication methods (no password in URI)
            uri = f"snowflake://{self.user}@{self.account}"

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

        # Add authentication-specific parameters
        if self.authenticator == "SNOWFLAKE_JWT":
            params.append(f"authenticator={self.authenticator}")
            if self.private_key_file:
                private_key_path = self._resolve_secret_value(self.private_key_file)
                params.append(f"private_key_file={private_key_path}")
            if self.private_key_file_pwd:
                private_key_pwd = self._resolve_secret_value(self.private_key_file_pwd)
                params.append(f"private_key_file_pwd={private_key_pwd}")
        elif self.authenticator == "OAUTH":
            params.append(f"authenticator={self.authenticator}")
            if self.oauth_client_id:
                client_id = self._resolve_secret_value(self.oauth_client_id)
                params.append(f"oauth_client_id={client_id}")
            if self.oauth_client_secret:
                client_secret = self._resolve_secret_value(self.oauth_client_secret)
                params.append(f"oauth_client_secret={client_secret}")
            if self.oauth_token_request_url:
                params.append(f"oauth_token_request_url={self.oauth_token_request_url}")
            if self.oauth_authorization_url:
                params.append(f"oauth_authorization_url={self.oauth_authorization_url}")

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
        if self.api_key and self.authenticator == "SNOWFLAKE":
            if resolved_api_key := self.api_key.resolve_value():
                masked_uri = masked_uri.replace(resolved_api_key, "***REDACTED***")

        # Mask private key password
        if self.private_key_file_pwd:
            private_key_pwd = self._resolve_secret_value(self.private_key_file_pwd)
            if private_key_pwd:
                masked_uri = masked_uri.replace(private_key_pwd, "***REDACTED***")

        # Mask OAuth client secret
        if self.oauth_client_secret:
            client_secret = self._resolve_secret_value(self.oauth_client_secret)
            if client_secret:
                masked_uri = masked_uri.replace(client_secret, "***REDACTED***")

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
