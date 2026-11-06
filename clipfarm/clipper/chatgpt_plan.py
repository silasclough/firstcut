"""Local Sign in with ChatGPT for the AI Videos editor.

This is an OSS-client implementation of OpenAI's documented PKCE flow.  It deliberately keeps credentials out of the
browser: the local server exposes only a safe connection state, while refresh credentials live in an AES-GCM file whose
key is stored in the macOS Keychain.  ChatGPT-plan inference uses the streaming Responses API with ``store: false``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import secrets
import subprocess
import threading
from html import escape
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from ..core.log import get_logger

log = get_logger("clipper.chatgpt_plan")
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
SHARING_SCOPE = "chatgpt.tokens.use.direct"
CALLBACK_PATH = "/auth/callback"
KEYCHAIN_SERVICE = "Clipfarm ChatGPT Plan Encryption Key"


def _crypto():
    """AES-GCM, imported only when ChatGPT sign-in is used: the server and the M2 never sign in, and don't need the package."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise ChatGPTPlanError("ChatGPT sign-in needs the 'cryptography' package (pip install -r requirements.txt)") from e
    return AESGCM


def _jwt():
    try:
        import jwt
    except ImportError as e:
        raise ChatGPTPlanError("ChatGPT sign-in needs the 'PyJWT' package (pip install -r requirements.txt)") from e
    return jwt


class ChatGPTPlanError(RuntimeError):
    """A deliberately credential-free error suitable for the local editor UI."""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _random() -> str:
    return _b64(secrets.token_bytes(32))


def _sse_text(raw: str | bytes) -> str:
    """Requests may leave SSE lines as bytes when the response omits a charset."""
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


class ChatGPTPlan:
    """One machine-local ChatGPT-plan connection. The object is safe to create for each LocalAI instance."""

    def __init__(self, cfg):
        data = Path(cfg.root) / cfg.get("clipper.ai_videos.data_dir", "data/ai_videos")
        self.dir = data / "chatgpt_plan"
        self.auth_file = self.dir / "chatgpt-auth.json"
        self.host_file = self.dir / "chatgpt-host.json"
        self.app_name = str(cfg.get("clipper.ai_videos.chatgpt_plan.app_name", "Clipfarm AI Videos"))
        self.model = str(cfg.get("clipper.ai_videos.chatgpt_plan.model", "gpt-5.4"))
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def save_usage(self, headers) -> None:
        """The plan's usage windows (x-codex-primary/secondary-*: used percent, window length, reset), for the app's sidebar. Skipped
        quietly when ChatGPT doesn't send them."""
        try:
            h = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
            out = {"at": time.time(), "status": None, "five_hour": None, "week": None}
            for which in ("primary", "secondary"):
                used = h.get(f"x-codex-{which}-used-percent")
                if used is None:
                    continue
                mins = float(h.get(f"x-codex-{which}-window-minutes") or (300 if which == "primary" else 10080))
                reset = h.get(f"x-codex-{which}-reset-at")
                after = h.get(f"x-codex-{which}-reset-after-seconds")
                resets = float(reset) if reset else time.time() + float(after) if after else 0.0
                out["five_hour" if mins <= 600 else "week"] = {"used": float(used) / 100, "resets": resets}
            if out["five_hour"] or out["week"]:
                self.dir.mkdir(parents=True, exist_ok=True)
                (self.dir / "usage.json").write_text(json.dumps(out))
        except (OSError, TypeError, ValueError):
            pass

    def usage(self) -> dict | None:
        try:
            return json.loads((self.dir / "usage.json").read_text())
        except (OSError, ValueError):
            return None

    # -- secure local state ---------------------------------------------------------------
    def _prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.dir.is_symlink():
            raise ChatGPTPlanError("ChatGPT credential storage must not be a symbolic link")
        if os.name != "nt":
            os.chmod(self.dir, 0o700)

    def _key(self) -> bytes:
        if platform.system() != "Darwin":
            raise ChatGPTPlanError("secure ChatGPT credential storage is currently configured for macOS only")
        account = os.environ.get("USER") or "clipfarm"
        find = subprocess.run(["security", "find-generic-password", "-a", account, "-s", KEYCHAIN_SERVICE, "-w"],
                              capture_output=True, text=True)
        secret = find.stdout.strip() if find.returncode == 0 else ""
        if not secret:
            secret = _b64(secrets.token_bytes(32))
            made = subprocess.run(["security", "add-generic-password", "-U", "-a", account, "-s", KEYCHAIN_SERVICE, "-w", secret],
                                  capture_output=True, text=True)
            if made.returncode:
                raise ChatGPTPlanError("macOS Keychain is unavailable; unlock it, then try ChatGPT sign-in again")
        try:
            key = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
        except ValueError as e:
            raise ChatGPTPlanError("the stored ChatGPT credential key is invalid; restore Keychain access") from e
        if len(key) != 32:
            raise ChatGPTPlanError("the stored ChatGPT credential key is invalid; restore Keychain access")
        return key

    def _put(self, path: Path, value: dict) -> None:
        self._prepare()
        tmp = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            os.chmod(path, 0o600)

    def _host_id(self) -> str:
        self._prepare()
        try:
            d = json.loads(self.host_file.read_text())
            value = d.get("id")
            if isinstance(value, str) and value.startswith("urn:uuid:") and str(uuid.UUID(value.removeprefix("urn:uuid:"))) == value.removeprefix("urn:uuid:"):
                return value
            raise ValueError
        except FileNotFoundError:
            value = f"urn:uuid:{uuid.uuid4()}"
            self._put(self.host_file, {"version": 1, "id": value})
            return value
        except (OSError, ValueError, json.JSONDecodeError) as e:
            # Versions before 2026-10-13 generated a malformed UUID and never completed
            # registration. Keep it as evidence, then replace it only when no credentials
            # exist; an established connection must never silently change host identity.
            if not self.auth_file.exists():
                rejected = self.host_file.with_name(f"chatgpt-host.invalid-{int(time.time())}.json")
                try:
                    os.replace(self.host_file, rejected)
                    value = f"urn:uuid:{uuid.uuid4()}"
                    self._put(self.host_file, {"version": 1, "id": value})
                    return value
                except OSError:
                    pass
            raise ChatGPTPlanError("saved ChatGPT host identity could not be read; restore chatgpt-host.json") from e

    def _read(self) -> dict | None:
        self._prepare()
        try:
            raw = json.loads(self.auth_file.read_text())
            cipher = base64.b64decode(raw["ciphertext"], validate=True)
            plain = _crypto()(self._key()).decrypt(cipher[:12], cipher[12:], b"clipfarm-chatgpt-plan-v1")
            value = json.loads(plain)
            if not isinstance(value, dict):
                raise ValueError
            return value
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001 - never reveal authentication material
            raise ChatGPTPlanError("saved ChatGPT credentials could not be decrypted; restore Keychain access or reconnect") from e

    def _write(self, value: dict) -> None:
        nonce = secrets.token_bytes(12)
        plain = json.dumps(value, separators=(",", ":")).encode()
        ciphertext = nonce + _crypto()(self._key()).encrypt(nonce, plain, b"clipfarm-chatgpt-plan-v1")
        self._put(self.auth_file, {"version": 1, "provider": "macos-keychain-aesgcm", "ciphertext": base64.b64encode(ciphertext).decode()})

    # -- OAuth -----------------------------------------------------------------------------
    def _discovery(self) -> dict:
        try:
            r = requests.get(f"{ISSUER}/.well-known/openid-configuration", timeout=20, allow_redirects=False)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            raise ChatGPTPlanError("could not reach ChatGPT sign-in") from e
        if not r.ok or d.get("issuer") != ISSUER or any(urlparse(str(d.get(k, ""))).scheme != "https" or urlparse(str(d.get(k, ""))).netloc != "auth.openai.com"
                                                     for k in ("authorization_endpoint", "token_endpoint", "jwks_uri")):
            raise ChatGPTPlanError("ChatGPT sign-in configuration could not be verified")
        return d

    def _verify_identity(self, token: str, client_id: str, nonce: str | None) -> dict:
        d = self._discovery()
        try:
            jwt = _jwt()
            key = jwt.PyJWKClient(d["jwks_uri"], timeout=15).get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=client_id, issuer=ISSUER,
                                options={"require": ["exp", "iat", "sub"]}, leeway=5)
        except Exception as e:  # noqa: BLE001
            raise ChatGPTPlanError("ChatGPT identity could not be verified; please sign in again") from e
        if not isinstance(claims.get("sub"), str) or (nonce is not None and claims.get("nonce") != nonce) or (claims.get("azp") is not None and claims["azp"] != client_id):
            raise ChatGPTPlanError("ChatGPT identity could not be verified; please sign in again")
        return claims

    def _token(self, endpoint: str, form: dict) -> dict:
        try:
            r = requests.post(endpoint, data=form, headers={"Accept": "application/json"}, timeout=45, allow_redirects=False)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            raise ChatGPTPlanError("ChatGPT sign-in could not complete; check your connection and try again") from e
        if not r.ok or not isinstance(d, dict):
            raise ChatGPTPlanError("ChatGPT sign-in did not return usable credentials; please try again")
        return d

    @staticmethod
    def _credentials(data: dict, old_scopes: list[str] | None = None) -> tuple[list[str], dict]:
        scopes = str(data.get("scope") or " ".join(old_scopes or [])).split()
        token, refresh, expires = data.get("access_token"), data.get("refresh_token"), data.get("expires_in")
        if not isinstance(token, str) or not isinstance(refresh, str) or not token or not refresh or not isinstance(expires, (int, float)) or expires <= 0:
            raise ChatGPTPlanError("ChatGPT did not return usable plan credentials; please sign in again")
        if SHARING_SCOPE not in scopes:
            raise ChatGPTPlanError("sign-in succeeded, but ChatGPT plan usage was not enabled")
        return scopes, {"access_token": token, "refresh_token": refresh, "expires_at": int(time.time() + float(expires))}

    def connect(self) -> dict:
        """Blocking browser sign-in. Call in a background thread, never from an HTTP handler."""
        if not self._lock.acquire(blocking=False):
            raise ChatGPTPlanError("a ChatGPT sign-in is already in progress")
        try:
            self._last_error = None
            previous = self._read()
            d = self._discovery()
            state, nonce, verifier = _random(), _random(), _random()
            received: dict[str, str] = {}
            done = threading.Event()

            class Callback(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):  # noqa: N802
                    parsed = urlparse(self.path)
                    q = parse_qs(parsed.query, keep_blank_values=True)
                    if parsed.path != CALLBACK_PATH or self.headers.get("Host") != f"127.0.0.1:{server.server_port}" or q.get("state") != [state]:
                        self.send_error(400, "Invalid sign-in state")
                        return
                    received.update({k: v[0] for k, v in q.items() if len(v) == 1})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    back = escape(self.server.app_name).encode()                     # ("FirstCut" in the app, else the Clipfarm page's name)
                    self.wfile.write(b"<!doctype html><title>Return to " + back + b"</title><h1>Return to " + back + b"</h1><p>Your ChatGPT connection is being verified. You can close this tab.</p>")
                    done.set()

            server = ThreadingHTTPServer(("127.0.0.1", 0), Callback)
            server.app_name = self.app_name
            server.timeout = 0.25
            try:
                redirect = f"http://127.0.0.1:{server.server_port}{CALLBACK_PATH}"
                challenge = _b64(hashlib.sha256(verifier.encode()).digest())
                client_id = previous.get("client_id") if previous else "dynamic_agent_client"
                query = {"client_id": client_id, "response_type": "code", "redirect_uri": redirect, "scope": SCOPES, "resource": RESOURCE,
                         "state": state, "nonce": nonce, "code_challenge_method": "S256", "code_challenge": challenge}
                if previous:
                    if previous.get("email"):
                        query["login_hint"] = previous["email"]
                    if previous.get("id_token"):
                        query["id_token_hint"] = previous["id_token"]
                else:
                    query["agent_name_hint"] = self.app_name
                    query["ext_agent_host_id"] = self._host_id()
                webbrowser.open_new(d["authorization_endpoint"] + "?" + urlencode(query))
                deadline = time.monotonic() + 600
                while not done.is_set() and time.monotonic() < deadline:
                    server.handle_request()
                if not done.is_set():
                    raise ChatGPTPlanError("ChatGPT sign-in timed out; please try again")
            finally:
                server.server_close()
            if received.get("error"):
                raise ChatGPTPlanError("ChatGPT plan usage was not enabled")
            code, issued = received.get("code"), received.get("client_id") or (previous or {}).get("client_id")
            if not code or not issued or issued == "dynamic_agent_client" or (previous and issued != previous.get("client_id")):
                raise ChatGPTPlanError("ChatGPT did not complete app registration; please try again")
            token = self._token(d["token_endpoint"], {"grant_type": "authorization_code", "client_id": issued, "code": code,
                                                         "code_verifier": verifier, "redirect_uri": redirect, "resource": RESOURCE})
            identity_token = token.get("id_token")
            if not isinstance(identity_token, str):
                raise ChatGPTPlanError("ChatGPT did not return a verifiable identity; please try again")
            claims = self._verify_identity(identity_token, issued, nonce)
            if previous and claims["sub"] != previous.get("subject"):
                raise ChatGPTPlanError("the sign-in returned a different ChatGPT account; add it as a separate connection later")
            scopes, credentials = self._credentials(token)
            self._write({"version": 1, "client_id": issued, "subject": claims["sub"], "email": claims.get("email", ""), "name": claims.get("name", ""),
                         "id_token": identity_token, "scopes": scopes, "credentials": credentials, "saved_at": int(time.time())})
            return self.status()
        except ChatGPTPlanError as e:
            self._last_error = str(e)
            raise
        except Exception as e:  # noqa: BLE001 - preserve the safe UI state but do not expose OAuth data
            self._last_error = "ChatGPT sign-in could not be completed; please try again"
            raise e
        finally:
            self._lock.release()

    # -- inference -------------------------------------------------------------------------
    def _connection(self) -> dict:
        with self._lock:
            saved = self._read()
            if not saved or SHARING_SCOPE not in saved.get("scopes", []) or not saved.get("credentials"):
                raise ChatGPTPlanError("connect ChatGPT plan usage before making AI requests")
            if int(saved["credentials"].get("expires_at", 0)) > int(time.time()) + 60:
                return saved
            d = self._discovery()
            data = self._token(d["token_endpoint"], {"grant_type": "refresh_token", "client_id": saved["client_id"],
                                                       "refresh_token": saved["credentials"].get("refresh_token", ""), "resource": RESOURCE})
            scopes, credentials = self._credentials(data, saved.get("scopes"))
            identity_token = data.get("id_token")
            if isinstance(identity_token, str):
                claims = self._verify_identity(identity_token, saved["client_id"], None)
                if claims["sub"] != saved["subject"]:
                    raise ChatGPTPlanError("the refreshed ChatGPT account did not match; please reconnect")
                saved["id_token"] = identity_token
                saved["email"], saved["name"] = claims.get("email", ""), claims.get("name", "")
            saved["scopes"], saved["credentials"], saved["saved_at"] = scopes, credentials, int(time.time())
            self._write(saved)
            return saved

    def status(self) -> dict:
        if self._lock.locked():
            return {"status": "connecting", "sharing": False}
        try:
            saved = self._read()
        except ChatGPTPlanError as e:
            return {"status": "error", "sharing": False, "error": str(e)}
        if not saved:
            return {"status": "error" if self._last_error else "disconnected", "sharing": False, **({"error": self._last_error} if self._last_error else {})}
        return {"status": "connected", "sharing": SHARING_SCOPE in saved.get("scopes", []) and bool(saved.get("credentials")),
                "email": saved.get("email") or None, "name": saved.get("name") or None, "model": self.model}

    def connected(self) -> bool:
        return bool(self.status().get("sharing"))

    def disconnect(self) -> None:
        """Remove local credentials. If remote revocation is unavailable, ChatGPT Settings remains the authority."""
        with self._lock:
            saved = self._read()
            if not saved:
                return
            try:
                endpoint = self._discovery().get("revocation_endpoint")
                refresh = (saved.get("credentials") or {}).get("refresh_token")
                if endpoint and refresh:
                    requests.post(endpoint, data={"token": refresh, "token_type_hint": "refresh_token", "client_id": saved.get("client_id", "")}, timeout=15)
            except Exception:  # noqa: BLE001 - local removal must still stop this editor using the connection
                log.warning("ChatGPT-plan remote revocation could not be confirmed; local credentials were removed")
            self.auth_file.unlink(missing_ok=True)

    def models(self) -> list[dict]:
        saved = self._connection()
        r = requests.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {saved['credentials']['access_token']}", "Accept": "application/json"}, timeout=30)
        try:
            data = r.json()
        except ValueError as e:
            raise ChatGPTPlanError("ChatGPT returned an invalid model list") from e
        models = data.get("models") if isinstance(data, dict) else None
        if not r.ok or not isinstance(models, list):
            raise ChatGPTPlanError("ChatGPT could not list the models available to this plan")
        return [{"slug": x["slug"], "display_name": x.get("display_name", x["slug"])} for x in models
                if isinstance(x, dict) and x.get("visibility") == "list" and isinstance(x.get("slug"), str)]

    def complete_json(self, system: str, prompt: str, schema: dict, model: str | None = None, images: list[bytes] | None = None,
                      reasoning_effort: str | None = None) -> tuple[dict, str]:
        saved = self._connection()
        available = self.models()
        wanted = model or self.model
        selected = next((x["slug"] for x in available if x["slug"] == wanted), available[0]["slug"] if available else None)
        if not selected:
            raise ChatGPTPlanError("no ChatGPT-plan model is available for this connection")
        instructions = system + "\n\nReturn only one JSON object matching this JSON Schema:\n" + json.dumps(schema, separators=(",", ":"))
        content: str | list[dict] = prompt
        if images:
            # Responses accepts the same image inputs as its ordinary API. If a particular plan/model does not admit vision,
            # OpenAI returns a capability error and we surface it rather than charging an API key behind the creator's back.
            content = [{"type": "input_text", "text": prompt}] + [
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}
                for image in images
            ]
        body = {"model": selected, "input": [{"role": "user", "content": content}], "instructions": instructions, "store": False, "stream": True}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        try:
            r = requests.post(f"{RESOURCE}/responses", json=body, headers={"Authorization": f"Bearer {saved['credentials']['access_token']}", "Accept": "text/event-stream"},
                              stream=True, timeout=(30, 300))
        except requests.RequestException as e:
            raise ChatGPTPlanError("could not reach ChatGPT for this editor request") from e
        self.save_usage(r.headers)
        if r.status_code == 429:
            raise ChatGPTPlanError("ChatGPT plan usage limit reached")
        if not r.ok:
            raise ChatGPTPlanError("ChatGPT-plan request was not accepted; check your plan usage limit or reconnect")
        text, event, lines, completed = "", "", [], False
        try:
            for raw in r.iter_lines(decode_unicode=False):
                raw = _sse_text(raw)
                if raw:
                    if raw.startswith("event:"):
                        event = raw[6:].strip()
                    elif raw.startswith("data:"):
                        lines.append(raw[5:].strip())
                    continue
                if not lines:
                    continue
                data = json.loads("\n".join(lines))
                lines = []
                typ = data.get("type", event) if isinstance(data, dict) else event
                if typ == "response.output_text.delta":
                    text += str(data.get("delta", ""))
                elif typ in ("response.failed", "response.incomplete", "error"):
                    raise ChatGPTPlanError("ChatGPT-plan request did not finish; try again or check usage limits")
                elif typ == "response.completed":
                    completed = True
                    break
        finally:
            r.close()
        if not completed:
            raise ChatGPTPlanError("ChatGPT-plan response ended before completion")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(cleaned)
        except ValueError as e:
            raise ChatGPTPlanError("ChatGPT returned invalid structured editor data; try again") from e
        if not isinstance(value, dict):
            raise ChatGPTPlanError("ChatGPT returned invalid structured editor data; try again")
        return value, selected
