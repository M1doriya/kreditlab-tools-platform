# Contributing

Thanks for your interest in `tensorlake-docai`. PRs are welcome.

## Expectations

Maintainer bandwidth is limited — response times are best-effort, often a
few business days. To keep things moving:

- Small, focused PRs land fastest. One change per PR.
- Big refactors or new model integrations: please open an issue first to
  align on the design before writing code.

## Dev setup

```bash
git clone https://github.com/tensorlakeai/openingest
cd openingest
pip install -e ".[dev]"
cp .env.example .env  # fill in only the keys you need
```

Run the smoke tests:

```bash
pytest tests/ -q
```

Format before committing:

```bash
black src/ tests/ examples/
ruff check src/ tests/ examples/
```

## PR checklist

- [ ] `pytest tests/ -q` passes
- [ ] New code has at least one test (mock external API clients)
- [ ] No new required env vars without a `.env.example` update + README mention
- [ ] No hardcoded keys, endpoints, or bucket names
- [ ] If you added a new OCR backend, document it in `docs/models.md`

## What we'll likely decline

- Adding a new ML provider integration without ongoing maintenance commitment
- Large architectural rewrites — the pipeline is intentionally simple
- Dependency bumps that pull in heavy new transitive deps
- PRs that mix unrelated changes

## Reporting bugs

Open a GitHub issue with: version (or commit SHA), what you ran, what you
expected, what happened, and a minimal reproducer.

For security issues, see [`SECURITY.md`](SECURITY.md) — don't file public
issues.

## License

By contributing you agree your contributions are licensed under
[Apache-2.0](LICENSE), the project license.
