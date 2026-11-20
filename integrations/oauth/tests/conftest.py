# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")


@pytest.fixture(scope="session")
def signing_material():
    """
    An RSA private key with its self-signed certificate, as PEM strings, for client-assertion tests.

    `thumbprint` is the expected `x5t#S256` header value, computed by `cryptography` rather than by the code under
    test.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "oauth-haystack-tests")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return SimpleNamespace(
        private_key_pem=key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode(),
        certificate_pem=certificate.public_bytes(serialization.Encoding.PEM).decode(),
        public_key=key.public_key(),
        thumbprint=base64.urlsafe_b64encode(certificate.fingerprint(hashes.SHA256())).rstrip(b"=").decode(),
    )
