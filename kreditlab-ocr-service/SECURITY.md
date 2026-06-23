# Security policy

## Reporting a vulnerability

If you discover a security vulnerability in `tensorlake-docai`, please report
it privately rather than opening a public GitHub issue.

**Email:** support@tensorlake.ai

Include enough detail for us to reproduce: affected version (or commit SHA),
steps to reproduce, and the impact you've observed. A proof-of-concept is
helpful but not required.

We'll acknowledge receipt within 3 business days and aim to provide a fix or
mitigation timeline within 14 days for confirmed issues.

## Scope

In scope:

- Code under `src/tensorlake_docai/` (including the deploy entrypoint
  `src/workflow.py`)
- Example code that demonstrates an unsafe pattern

Out of scope:

- Bugs in upstream dependencies — report those to their maintainers
- Vulnerabilities in the Tensorlake hosted platform — those go to
  support@tensorlake.ai with the subject prefix `[platform]` so they reach the
  right team
- Findings that require credentials or keys you weren't given

## Supported versions

Only the `main` branch receives security fixes; please update to the latest
commit before reporting.

## Disclosure

We follow coordinated disclosure: fixes ship first, then we publish an
advisory. Reporters who follow this policy will be credited in the advisory
unless they prefer to remain anonymous.
