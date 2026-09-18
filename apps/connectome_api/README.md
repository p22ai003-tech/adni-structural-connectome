# Connectome API

Read-only FastAPI layer over the existing connectome analysis artifacts.

The API never writes to the scientific data tree. During migration, the
Streamlit service remains the reference application.

Development:

```bash
cd /home/ec2-user/exp
./.venv_connectome_api/bin/uvicorn apps.connectome_api.main:app \
  --host 127.0.0.1 --port 8001
```

OpenAPI is served at `/api/docs`; versioned routes are under `/api/v1`.
The React client regenerates its checked TypeScript contract from
`/api/openapi.json` with `npm run generate:api-types`.
