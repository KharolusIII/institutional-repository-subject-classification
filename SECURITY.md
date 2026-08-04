# Security and responsible disclosure

## Public-data boundary

This repository contains source code, synthetic examples, aggregate research
results, and publication figures. It must not contain repository exports,
full text, item-level predictions, frozen manifests, model checkpoints,
credentials, access tokens, or personal filesystem paths.

Use environment variables or Colab Secrets for credentials. Never place a
token in a Git URL, notebook output, YAML file, log, or Git remote.

## Reporting a security issue

Please report suspected credential exposure privately to the repository owner
through GitHub's private vulnerability reporting facility. Do not open a
public issue containing a secret or personal data.

If a credential is exposed, revoke it immediately. Removing it from the latest
commit is not sufficient because Git history may retain it.
