# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Dict, Final, Optional, Union

import pandas as pd
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport("Run 'pip install snowflake-connector-python>=3.10.1'") as snow_import:
    import snowflake.connector
    from snowflake.connector.connection import SnowflakeConnection
    from snowflake.connector.errors import (
        DatabaseError,
        ForbiddenError,
        ProgrammingError,
    )

logger = logging.getLogger(__name__)

MAX_SYS_ROWS: Final = 1000000  # Max rows to fetch from a table


@component
class SnowflakeTableRetriever:
    """
    Connects to a Snowflake database to execute a SQL query.
    For more information, see [Snowflake documentation](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector).

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

    # When database and schema are provided during component initialization.
    query = "SELECT * FROM table_name"

    # or

    # When database and schema are NOT provided during component initialization.
    query = "SELECT * FROM database_name.schema_name.table_name"

    results = executor.run(query=query)

    print(results["dataframe"].head(2))  # Pandas dataframe
    #   Column 1  Column 2
    # 0       Value1 Value2
    # 1       Value1 Value2

    print(results["table"])  # Markdown
    # | Column 1  | Column 2  |
    # |:----------|:----------|
    # | Value1    | Value2    |
    # | Value1    | Value2    |
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
        login_timeout: Optional[int] = None,
    ) -> None:
        """
        :param user: User's login.
        :param account: Snowflake account identifier.
        :param api_key: Snowflake account password.
        :param database: Name of the database to use.
        :param db_schema: Name of the schema to use.
        :param warehouse: Name of the warehouse to use.
        :param login_timeout: Timeout in seconds for login. By default, 60 seconds.
        """

        self.user = user
        self.account = account
        self.api_key = api_key
        self.database = database
        self.db_schema = db_schema
        self.warehouse = warehouse
        self.login_timeout = login_timeout or 60

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            user=self.user,
            account=self.account,
            api_key=self.api_key.to_dict(),
            database=self.database,
            db_schema=self.db_schema,
            warehouse=self.warehouse,
            login_timeout=self.login_timeout,
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

    @staticmethod
    def _snowflake_connector(connect_params: Dict[str, Any]) -> Union[SnowflakeConnection, None]:
        """
        Connect to a Snowflake database.

        :param connect_params: Snowflake connection parameters.
        """
        try:
            return snowflake.connector.connect(**connect_params)
        except DatabaseError as e:
            logger.error("{error_msg} ", errno=e.errno, error_msg=e.msg)
            return None

    @staticmethod
    def _extract_table_names(query: str) -> list:
        """
        Extract table names from an SQL query using regex.
        The extracted table names will be checked for privilege.

        :param query: SQL query to extract table names from.
        """

        suffix = "\\s+([a-zA-Z0-9_.]+)"  # Regular expressions to match table names in various clauses

        patterns = [
            "MERGE\\s+INTO",
            "USING",
            "JOIN",
            "FROM",
            "UPDATE",
            "DROP\\s+TABLE",
            "TRUNCATE\\s+TABLE",
            "CREATE\\s+TABLE",
            "INSERT\\s+INTO",
            "DELETE\\s+FROM",
        ]

        # Combine all patterns into a single regex
        combined_pattern = "|".join([pattern + suffix for pattern in patterns])

        # Find all matches in the query
        matches = re.findall(pattern=combined_pattern, string=query, flags=re.IGNORECASE)

        # Flatten the list of tuples and remove duplication
        matches = list(set(sum(matches, ())))

        # Clean and return unique table names
        return [match.strip('`"[]').upper() for match in matches if match]

    @staticmethod
    def _execute_sql_query(conn: SnowflakeConnection, query: str) -> pd.DataFrame:
        """
        Execute an SQL query and fetch the results.

        :param conn: An open connection to Snowflake.
        :param query: The query to execute.
        """
        cur = conn.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchmany(size=MAX_SYS_ROWS)  # set a limit to avoid fetching too many rows

            df = pd.DataFrame(rows, columns=[desc.name for desc in cur.description])  # Convert data to a dataframe
            return df
        except Exception as e:
            if isinstance(e, ProgrammingError):
                logger.warning(
                    "{error_msg} Use the following ID to check the status of the query in Snowflake UI (ID: {sfqid})",
                    error_msg=e.msg,
                    sfqid=e.sfqid,
                )
            else:
                logger.warning("An unexpected error occurred: {error_msg}", error_msg=e)

        return pd.DataFrame()

    @staticmethod
    def _has_select_privilege(privileges: list, table_name: str) -> bool:
        """
        Check user's privilege for a specific table.

        :param privileges: List of privileges.
        :param table_name: Name of the table.
        """

        for privilege in reversed(privileges):
            if table_name.lower() == privilege[3].lower() and re.match(
                pattern="truncate|update|insert|delete|operate|references",
                string=privilege[1],
                flags=re.IGNORECASE,
            ):
                return False

        return True

    def _check_privilege(
        self,
        conn: SnowflakeConnection,
        query: str,
        user: str,
    ) -> bool:
        """
        Check whether a user has a `select`-only access to the table.

        :param conn: An open connection to Snowflake.
        :param query: The query from where to extract table names to check read-only access.
        """
        cur = conn.cursor()

        cur.execute(f"SHOW GRANTS TO USER {user};")

        # Get user's latest role
        roles = cur.fetchall()
        if not roles:
            logger.error("User does not exist")
            return False

        # Last row second column from GRANT table
        role = roles[-1][1]

        # Get role privilege
        cur.execute(f"SHOW GRANTS TO ROLE {role};")

        # Keep table level privileges
        table_privileges = [row for row in cur.fetchall() if row[2] == "TABLE"]

        # Get table names to check for privilege
        table_names = self._extract_table_names(query=query)

        for table_name in table_names:
            if not self._has_select_privilege(
                privileges=table_privileges,
                table_name=table_name,
            ):
                return False
        return True

    def _fetch_data(
        self,
        query: str,
    ) -> pd.DataFrame:
        """
        Fetch data from a database using a SQL query.

        :param query: SQL query to use to fetch the data from the database. Query must be a valid SQL query.
        """

        df = pd.DataFrame()
        if not query:
            return df
        try:
            # Create a new connection with every run
            conn = self._snowflake_connector(
                connect_params={
                    "user": self.user,
                    "account": self.account,
                    "password": self.api_key.resolve_value(),
                    "database": self.database,
                    "schema": self.db_schema,
                    "warehouse": self.warehouse,
                    "login_timeout": self.login_timeout,
                }
            )
            if conn is None:
                return df
        except (ForbiddenError, ProgrammingError) as e:
            logger.error(
                "Error connecting to Snowflake ({errno}): {error_msg}",
                errno=e.errno,
                error_msg=e.msg,
            )
            return df

        try:
            # Check if user has `select` privilege on the table
            if self._check_privilege(
                conn=conn,
                query=query,
                user=self.user,
            ):
                df = self._execute_sql_query(conn=conn, query=query)
            else:
                logger.error("User does not have `Select` privilege on the table.")

        except Exception as e:
            logger.error("An unexpected error has occurred: {error}", error=e)

        # Close connection after every execution
        conn.close()
        return df

    @component.output_types(dataframe=pd.DataFrame, table=str)
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute a SQL query against a Snowflake database.

        :param query: A SQL query to execute.
        """
        if not query:
            logger.error("Provide a valid SQL query.")
            return {
                "dataframe": pd.DataFrame,
                "table": "",
            }
        else:
            df = self._fetch_data(query)
            table_markdown = df.to_markdown(index=False) if not df.empty else ""

        return {"dataframe": df, "table": table_markdown}
