# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import uuid
from collections.abc import Callable, Iterator

import pytest

from haystack_integrations.document_stores.solr import SolrDocumentStore

SOLR_URL = os.environ.get("SOLR_URL", "http://localhost:8983/solr")

#: Dimension used by `haystack.testing.document_store.create_filterable_docs`.
#:
#: Every test must use this one value. Test cores are created from the shared `_default` configset, and
#: such cores reference the configset's managed schema instead of owning one, so the `embedding` field
#: is shared across all of them. A store built with a different dimension would write a conflicting
#: field type into that shared schema and every later test would be refused.
TEST_EMBEDDING_DIM = 768


def _unique_core_name() -> str:
    """Give every test its own core, so that tests can run in parallel without interfering."""
    return f"test_{uuid.uuid4().hex}"


def _drop_core(store: SolrDocumentStore) -> None:
    """Unload a core and delete its data directory."""
    with contextlib.suppress(Exception):
        store._solr_client.request(
            "GET",
            "admin/cores",
            params={
                "action": "UNLOAD",
                "core": store._core,
                "deleteIndex": True,
                "deleteInstanceDir": True,
            },
        )


@pytest.fixture
def solr_store() -> Iterator[Callable[..., SolrDocumentStore]]:
    """
    Factory for `SolrDocumentStore`s backed by a throwaway core.

    Assumes Solr is reachable at `SOLR_URL` - run `docker compose up -d --wait` from
    `integrations/solr` to bring it up.
    """
    created: list[SolrDocumentStore] = []

    def _make(**overrides) -> SolrDocumentStore:
        kwargs = {
            "url": SOLR_URL,
            "core": _unique_core_name(),
            "embedding_dim": TEST_EMBEDDING_DIM,
            "create_core": True,
            # The container runs without authentication, and the default would otherwise pick up
            # SOLR_USERNAME/SOLR_PASSWORD from the developer's environment.
            "auth": None,
            **overrides,
        }
        store = SolrDocumentStore(**kwargs)
        created.append(store)
        return store

    yield _make

    for store in created:
        _drop_core(store)
        store.close()
        with contextlib.suppress(Exception):
            asyncio.run(store.close_async())


@pytest.fixture
def document_store(solr_store: Callable[..., SolrDocumentStore]) -> SolrDocumentStore:
    """
    A store on a fresh core, used by the shared document store test suites.

    Embeddings are returned because those suites compare whole documents, embedding included, against
    what they wrote - which also makes them cover the vector round-trip.
    """
    return solr_store(return_embedding=True)


@pytest.fixture
def document_store_no_embedding_returned(
    solr_store: Callable[..., SolrDocumentStore],
) -> SolrDocumentStore:
    """A store left at the default `return_embedding=False`."""
    return solr_store(return_embedding=False)
