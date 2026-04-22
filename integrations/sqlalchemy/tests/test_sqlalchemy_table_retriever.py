# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

import haystack_integrations.components.retrievers.sqlalchemy.sqlalchemy_table_retriever as module
from haystack_integrations.components.retrievers.sqlalchemy import SQLAlchemyTableRetriever


class TestSQLAlchemyTableRetrieverInit:
    def test_init_defaults(self):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite")
        assert retriever.drivername == "sqlite"
        assert retriever.username is None
        assert retriever.password is None
        assert retriever.host is None
        assert retriever.port is None
        assert retriever.database is None
        assert retriever.init_script is None

    def test_init_all_params(self):
        password = Secret.from_token("secret")
        retriever = SQLAlchemyTableRetriever(
            drivername="postgresql+psycopg2",
            username="user",
            password=password,
            host="localhost",
            port=5432,
            database="mydb",
            init_script=["CREATE TABLE t (x INTEGER)"],
        )
        assert retriever.drivername == "postgresql+psycopg2"
        assert retriever.username == "user"
        assert retriever.password is password
        assert retriever.host == "localhost"
        assert retriever.port == 5432
        assert retriever.database == "mydb"
        assert retriever.init_script == ["CREATE TABLE t (x INTEGER)"]


class TestSQLAlchemyTableRetrieverSerialization:
    def test_to_dict(self):
        password = Secret.from_env_var("DB_PASSWORD")
        init_script = ["CREATE TABLE t (x INTEGER)", "INSERT INTO t VALUES (1)"]
        retriever = SQLAlchemyTableRetriever(
            drivername="sqlite",
            database=":memory:",
            password=password,
            init_script=init_script,
        )
        d = retriever.to_dict()
        expected_type = (
            "haystack_integrations.components.retrievers.sqlalchemy.sqlalchemy_table_retriever.SQLAlchemyTableRetriever"
        )
        assert d["type"] == expected_type
        params = d["init_parameters"]
        assert params["drivername"] == "sqlite"
        assert params["database"] == ":memory:"
        assert params["password"]["type"] == "env_var"
        assert params["init_script"] == init_script

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("DB_PASSWORD", "secret")
        password = Secret.from_env_var("DB_PASSWORD")
        init_script = ["CREATE TABLE t (x INTEGER)", "INSERT INTO t VALUES (1)"]
        retriever = SQLAlchemyTableRetriever(
            drivername="sqlite",
            database=":memory:",
            password=password,
            init_script=init_script,
        )
        d = retriever.to_dict()
        restored = SQLAlchemyTableRetriever.from_dict(d)
        assert restored.drivername == "sqlite"
        assert restored.database == ":memory:"
        assert restored.password is not None
        assert restored.password.resolve_value() == "secret"
        assert restored.init_script == init_script


class TestSQLAlchemyTableRetrieverRun:
    @pytest.fixture()
    def retriever_with_data(self):
        init_sql = [
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL,"
            " department TEXT NOT NULL, salary INTEGER NOT NULL)",
            "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000)",
            "INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 72000)",
            "INSERT INTO employees VALUES (3, 'Carol', 'Engineering', 88000)",
        ]
        retriever = SQLAlchemyTableRetriever(
            drivername="sqlite",
            database=":memory:",
            init_script=init_sql,
        )
        retriever.warm_up()
        return retriever

    def test_run_non_string_query(self):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
        result = retriever.run(query=123)  # type: ignore[arg-type]
        assert result["error"] == "query is not a string"
        assert result["dataframe"].empty
        assert result["table"] == ""

    @pytest.mark.parametrize("query", ["", "   ", "\t", "\n"])
    def test_run_empty_query(self, query):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
        result = retriever.run(query=query)
        assert result["error"] == "empty query"
        assert result["dataframe"].empty
        assert result["table"] == ""

    def test_run_returns_dataframe(self, retriever_with_data):
        result = retriever_with_data.run(query="SELECT * FROM employees ORDER BY id")
        df = result["dataframe"]
        assert result["error"] == ""
        assert list(df.columns) == ["id", "name", "department", "salary"]
        assert len(df) == 3
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["department"] == "Marketing"
        assert df.iloc[2]["salary"] == 88000

    def test_run_returns_markdown(self, retriever_with_data):
        result = retriever_with_data.run(query="SELECT * FROM employees ORDER BY id")
        expected = (
            "| id | name | department | salary |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Alice | Engineering | 95000 |\n"
            "| 2 | Bob | Marketing | 72000 |\n"
            "| 3 | Carol | Engineering | 88000 |"
        )
        assert result["table"] == expected

    def test_run_sql_error(self):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
        retriever.warm_up()
        result = retriever.run(query="SELECT * FROM nonexistent_table_xyz")
        assert result["error"] != ""
        assert result["dataframe"].empty

    def test_max_row_limit(self, monkeypatch):
        init_sql = [
            "CREATE TABLE t (x INTEGER)",
            "INSERT INTO t VALUES (1)",
            "INSERT INTO t VALUES (2)",
            "INSERT INTO t VALUES (3)",
        ]
        retriever = SQLAlchemyTableRetriever(
            drivername="sqlite",
            database=":memory:",
            init_script=init_sql,
        )
        monkeypatch.setattr(module, "MAX_SYS_ROWS", 2)
        retriever.warm_up()
        result = retriever.run(query="SELECT * FROM t")
        assert len(result["dataframe"]) == 2

    def test_warm_up_idempotent(self):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
        retriever.warm_up()
        engine = retriever._engine
        retriever.warm_up()
        assert retriever._engine is engine

    def test_run_calls_warm_up_automatically(self):
        retriever = SQLAlchemyTableRetriever(drivername="sqlite", database=":memory:")
        assert not retriever._warmed_up
        result = retriever.run(query="SELECT 1 AS x")
        assert retriever._warmed_up
        assert not result["dataframe"].empty
