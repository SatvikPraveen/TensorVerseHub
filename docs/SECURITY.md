# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x (latest) | ✅ |
| < 1.0 | ❌ |

---

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Report security issues privately by emailing **contact@tensorversehub.com**. Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

You will receive an acknowledgement within **48 hours** and a detailed response within **7 days**.

---

## Scope

TensorVerseHub is primarily a **learning and reference project**. Security considerations apply to:

- The Flask REST API (`examples/serving_examples/flask_tensorflow_api.py`)
- The FastAPI serving example
- Docker image dependencies
- Any user-facing input handling in examples

Out of scope:
- Vulnerabilities in third-party libraries (report to the respective maintainer)
- Issues that require physical access to the deployment machine

---

## Known Security Considerations

1. **Model serving in production**: The Flask example disables FLASK_DEBUG and uses proper error handling. For production deployments, add authentication, rate limiting, and HTTPS.
2. **Dependency pinning**: `requirements.txt` pins all versions. Update regularly with `pip list --outdated` and audit with `pip audit`.
3. **Docker non-root user**: The Dockerfile runs the application as a non-root `tensorverse` user.
4. **No credentials in source**: Never commit API keys, tokens, or passwords. Use environment variables or secrets management.
