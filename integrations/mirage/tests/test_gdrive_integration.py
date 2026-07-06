# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Live Google Drive integration test.

Mirage's Google Drive backend only accepts OAuth *user* credentials (`client_id` + `refresh_token`)
and mints access tokens via the refresh-token grant (see `mirage.core.google._client.TokenManager`).
Obtaining a durable refresh token requires publishing and verifying an *external* OAuth app, which is
impractical for CI. Instead, wer provision a Google service account (the `GOOGLE_DRIVE_SA_KEY`
secret), and CI mints a short-lived `drive.readonly` *access token* from it, exposed to the test run as
`GOOGLE_DRIVE_ACCESS_TOKEN` (see `.github/workflows/mirage.yml`).

Mirage 0.0.3 has no field for a pre-minted access token, so we substitute *only* the token-minting step:
`mirage.core.google._client.refresh_access_token` — the single choke point behind `TokenManager` — is
patched to return the service-account access token instead of performing a refresh-token exchange. Every
real Drive API call still goes out over the network with that bearer token, so this remains a genuine
end-to-end integration test; only the credential *type* is adapted.

If Mirage later accepts a pre-minted access token directly in `GoogleConfig`, this patch can be dropped
in favor of passing the token through the mount config.
"""

import os

import mirage.core.google._client as gclient
import pytest
from haystack.utils import Secret

from haystack_integrations.tools.mirage import MirageMount, MirageShellTool, MirageWorkspace


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN"),
    reason="GOOGLE_DRIVE_ACCESS_TOKEN not set",
)
class TestLiveGoogleDrive:
    @pytest.fixture(autouse=True)
    def _seed_service_account_token(self, monkeypatch):
        """
        Feed Mirage the service-account access token in place of a refresh-token exchange.

        `TokenManager.get_token` looks up `refresh_access_token` as a module global at call time, so
        patching it here makes every Drive request use the pre-minted token. The real HTTP calls are
        untouched, so the test still exercises the live Drive API.
        """
        token = os.environ["GOOGLE_DRIVE_ACCESS_TOKEN"]

        async def _return_service_account_token(*_args, **_kwargs):
            # (access_token, expires_in_seconds); the SA token is valid for ~1h, ample for a test.
            return token, 3600

        monkeypatch.setattr(gclient, "refresh_access_token", _return_service_account_token)

    @staticmethod
    def _workspace() -> MirageWorkspace:
        # client_id / refresh_token are never used (token minting is patched above), but Mirage's
        # GoogleDriveConfig requires them to construct, so pass placeholders.
        return MirageWorkspace(
            mounts=[
                MirageMount(
                    path="/gdrive",
                    resource="gdrive",
                    config={"client_id": "unused", "refresh_token": Secret.from_token("unused")},
                    read_only=True,
                )
            ]
        )

    def test_workspace_lists_drive_root(self):
        ws = self._workspace()
        try:
            out = ws.run("ls /gdrive")
        finally:
            ws.close()
        # A successful readdir exits 0; an auth/API failure would append "[exit code N]" (see _to_text).
        # Content is not pinned because what the service account can see (its own Drive plus files shared
        # with it) varies, so we assert the round-trip succeeded rather than specific file names.
        assert isinstance(out, str)
        assert "[exit code" not in out

    def test_shell_tool_lists_drive_root(self):
        # Exercise the actual product surface: an allowlisted MirageShellTool over the Drive mount.
        ws = self._workspace()
        tool = MirageShellTool(ws, allowed_commands=["ls"])
        try:
            out = tool._invoke("ls /gdrive")
        finally:
            tool.close()
        assert isinstance(out, str)
        assert "[exit code" not in out
