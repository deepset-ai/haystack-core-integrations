import asyncio

import pytest


@pytest.mark.asyncio
async def test_async_stub():
    # Arrange
    await asyncio.sleep(0)

    # Act
    result = "ok"

    # Assert
    assert result == "ok"
