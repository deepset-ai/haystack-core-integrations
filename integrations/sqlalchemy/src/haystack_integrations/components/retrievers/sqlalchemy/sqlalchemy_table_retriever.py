# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from pandas import DataFrame

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

MAX_SYS_ROWS = 10_000


@component
class SQLAlchemyTableRetriever:
    """
    Connects to any SQLAlchemy-supported database and executes a SQL query.

    Returns results as a Pandas DataFrame and an optional Markdown-formatted table string.
    Supports any database backend that SQLAlchemy supports, including PostgreSQL, MySQL,
    SQLite, and MSSQL.

    ### Usage example:

    ```python
    from haystack_integrations.components.retrievers.sqlalchemy import SQLAlchemyTableRetriever

    retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
    retriever.warm_up()
    result = retriever.run(query="SELECT 1 AS value")
    print(result["dataframe"])
    print(result["table"])
    ```
    """

    def __init__(
        self,
        drivername: str,
        username: str | None = None,
        password: Secret | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        init_script: list[str] | None = None,
    ) -> None:
        """
        Initialize SQLAlchemyTableRetriever.

        :param drivername: The SQLAlchemy driver name (e.g., ``"sqlite"``,
            ``"postgresql+psycopg2"``).
        :param username: Database username.
        :param password: Database password as a Haystack ``Secret``.
        :param host: Database host.
        :param port: Database port.
        :param database: Database name or path (e.g., ``":memory:"`` for SQLite in-memory).
        :param init_script: Optional list of SQL statements executed once on ``warm_up()``
            (e.g., to create tables or insert seed data). Each statement should be a
            separate string in the list.
        """
        self.drivername = drivername
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.init_script = init_script
        self._engine: Engine | None = None
        self._warmed_up = False

    def warm_up(self) -> None:
        """
        Initialize the database engine and execute ``init_script`` if provided.

        Called automatically by ``run()`` on first invocation if not already warmed up.
        """
        if self._warmed_up:
            return

        url = URL.create(
            drivername=self.drivername,
            username=self.username,
            password=self.password.resolve_value() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.database,
        )

        engine_kwargs: dict[str, Any] = {}
        if url.drivername.startswith("sqlite") and url.database in (":memory:", "", None):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool

        self._engine = create_engine(url, **engine_kwargs)

        if self.init_script:
            with self._engine.connect() as conn:
                for stmt in self.init_script:
                    stripped = stmt.strip()
                    if stripped:
                        conn.execute(text(stripped))
                conn.commit()

        self._warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            drivername=self.drivername,
            username=self.username,
            password=self.password.to_dict() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.database,
            init_script=self.init_script,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SQLAlchemyTableRetriever":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data.get("init_parameters", {}), ["password"])
        return default_from_dict(cls, data)

    @staticmethod
    def _df_to_markdown(df: DataFrame) -> str:
        if df.empty:
            return ""
        header = "| " + " | ".join(str(c) for c in df.columns) + " |"
        separator = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([header, separator, *rows])

    @component.output_types(dataframe=DataFrame, table=str, error=str)
    def run(self, query: str) -> dict[str, Any]:
        """
        Execute a SQL query and return the results.

        :param query: The SQL query to execute.
        :returns: A dictionary with:

            - ``dataframe``: A Pandas DataFrame with the query results.
            - ``table``: A Markdown-formatted string of the results.
            - ``error``: An error message if the query failed, otherwise an empty string.
        """
        if not isinstance(query, str):
            logger.warning("Query is not a string, returning empty DataFrame")
            return {"dataframe": DataFrame(), "table": "", "error": "query is not a string"}

        if not query.strip():
            return {"dataframe": DataFrame(), "table": "", "error": "empty query"}

        if not self._warmed_up:
            self.warm_up()

        if self._engine is None:  # pragma: no cover
            msg = "Engine is not initialized."
            raise RuntimeError(msg)

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchmany(MAX_SYS_ROWS)
                columns = list(result.keys())
            df = DataFrame(rows, columns=columns)
            return {"dataframe": df, "table": self._df_to_markdown(df), "error": ""}
        except SQLAlchemyError as e:
            logger.warning("Error executing query: {error}", error=str(e))
            return {"dataframe": DataFrame(), "table": "", "error": str(e)}
