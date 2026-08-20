#!/usr/bin/env python3
# web/security.py

"""
Security helpers for the web interface.

Description:
URL vetting (anti-SSRF) and a small optional shared-token gate. Kept separate
from app.py so the checks are unit-testable without spinning up the server.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import hmac
import ipaddress
import os
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Runtime knobs (env-driven, safe defaults)
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off", ""}


# Shared token. Unset -> auth disabled (local use, unchanged behaviour).
AUTH_TOKEN: Optional[str] = os.environ.get("WHISPR_AUTH_TOKEN") or None

# Upload ceiling (streamed, so this really caps disk/RAM).
MAX_UPLOAD_BYTES: int = _int_env("WHISPR_MAX_UPLOAD_MB", 500) * 1024 * 1024

# Hard cap on parallel workers a request may ask for.
MAX_WORKERS: int = _int_env("WHISPR_MAX_WORKERS", 8)

# Let operators kill the remote-fetch feature entirely (drops SSRF surface).
ALLOW_URL_FETCH: bool = _bool_env("WHISPR_ALLOW_URL_FETCH", True)


# ---------------------------------------------------------------------------
# Token check (constant-time)
# ---------------------------------------------------------------------------

def token_ok(supplied: Optional[str]) -> bool:
    """True when auth is off, or the supplied token matches the configured one."""
    if AUTH_TOKEN is None:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(supplied, AUTH_TOKEN)


def token_from_headers(headers) -> Optional[str]:
    """Pull a token from X-Whispr-Token or an Authorization: Bearer header."""
    token = headers.get("x-whispr-token")
    if token:
        return token
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return None


# ---------------------------------------------------------------------------
# Anti-SSRF URL vetting
# ---------------------------------------------------------------------------

def is_safe_remote_url(url: str) -> Tuple[bool, str]:
    """
    Decide whether a user-supplied URL is safe to fetch server-side.

    Rejects anything that isn't plain http/https or that resolves to a
    non-public address (loopback, RFC1918, link-local incl. the cloud
    metadata 169.254.0.0/16, reserved, multicast, unspecified).

    Returns (ok, reason). reason is a short, non-leaky string on rejection.

    Note: this resolves at check time; a TOCTOU/DNS-rebinding attacker could
    still flip the record before the fetch. Behind an egress-restricted
    network (see README) that residual is closed off too.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "malformed URL"

    if parsed.scheme not in {"http", "https"}:
        return False, "only http/https URLs are allowed"

    host = parsed.hostname
    if not host:
        return False, "URL has no host"

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, "host does not resolve"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False, "unresolvable address"
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, "URL resolves to a non-public address"

    return True, ""


# ---------------------------------------------------------------------------
# Upload filename hardening
# ---------------------------------------------------------------------------

def safe_upload_name(filename: Optional[str]) -> str:
    """
    Reduce an uploaded filename to a bare, single-segment basename.

    Kills path traversal and absolute paths: only the last path component is
    kept, separators/NULs stripped, and '.'/'..' rejected. The extension is
    preserved so backends can still sniff the format.
    """
    raw = (filename or "").replace("\x00", "")
    # Treat both separators as separators regardless of host OS.
    base = os.path.basename(raw.replace("\\", "/")).strip()
    if not base or base in {".", ".."}:
        base = "upload"
    return base[:255]
