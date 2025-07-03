import pytest


@pytest.mark.integration
def test_delete_table_first_call(document_store):
    """
    should be able to call _ensure_db_setup() inside itself while the connection is not yet established, and
    should not throw any exceptions.
    """
    document_store.delete_table()  # if throw error, test fails


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_table_async_first_call(document_store):
    """
    should be able to call _ensure_db_setup_async() inside itself while the connection is not yet established, and
    should not throw any exceptions.
    """
    await document_store.delete_table_async()  # if throw error, test fails
