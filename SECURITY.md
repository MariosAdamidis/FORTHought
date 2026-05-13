# Security

FORTHought is designed as a local/private-network research platform. Do not expose services directly to the public internet.

## Do Not Commit

- API keys, JWTs, database credentials, cookies, or tunnel credentials
- Real `.env` files (use `.env.example` as template)
- Production Docker Compose files with tunnel or proxy configuration
- User profiles, model grants, private prompts, traces, or chat exports
- Uploaded lab data or generated files containing private data

## Deployment

- Bind local services to `127.0.0.1` unless you deliberately configure otherwise.
- If you use a reverse proxy or tunnel, keep that configuration outside Git.
- Keep paper retrieval open-access-only in shared code.
- Treat Git history as public once pushed.

## Reporting

For security-sensitive issues, contact the repository owner directly rather than opening a public issue.
