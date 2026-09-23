# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x (latest) | ✅ |
| < 2.0 | ❌ |

---

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue with exploit details for security vulnerabilities.**

Report security issues privately through GitHub's **private vulnerability reporting** on the
[repository's Security tab](https://github.com/SatvikPraveen/TensorVerseHub/security/advisories/new).
If that is unavailable to you, open a [GitHub issue](https://github.com/SatvikPraveen/TensorVerseHub/issues)
that describes the affected area without exploit details and the maintainer will follow up privately. Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

You will receive an acknowledgement within **48 hours** and a detailed response within **7 days**.

---

## Scope

TensorVerseHub is primarily a **learning and reference project**. Security considerations apply to:

- The FastAPI model server (`tensorverse serve`, `tensorversehub/cli/serve.py`)
- The Flask REST API example (`examples/serving_examples/flask_tensorflow_api.py`)
- The Streamlit demo (`examples/serving_examples/streamlit_tensorflow_demo.py`)
- Docker image dependencies
- Any user-facing input handling in examples

Out of scope:
- Vulnerabilities in third-party libraries (report to the respective maintainer)
- Issues that require physical access to the deployment machine

---

## Known Security Considerations

1. **Model serving in production**: `tensorverse serve` and the Flask example ship without authentication, rate limiting or TLS. For production deployments put them behind a reverse proxy and add authentication, rate limiting and HTTPS.
2. **Dependency management**: `pyproject.toml` declares version ranges; `requirements.txt` pins the full notebook environment. Update regularly with `pip list --outdated` and audit with `pip-audit`.
3. **Docker non-root user**: All image targets (`runtime`, `api`, `jupyter`) run as the non-root `tensorverse` user.
4. **No credentials in source**: Never commit API keys, tokens, or passwords. Use environment variables or secrets management. `pre-commit` runs `detect-private-key` on every commit.
