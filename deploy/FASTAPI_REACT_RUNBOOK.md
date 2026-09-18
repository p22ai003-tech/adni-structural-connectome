# FastAPI + React shadow runbook

## Service boundaries

- Streamlit reference: `127.0.0.1:8501`, public HTTPS root `/`
- FastAPI/React shadow: `127.0.0.1:8001`
- React preview through nginx: `/next/`
- Versioned API and OpenAPI: `/api/v1/*`, `/api/docs`

The nginx server-level basic authentication protects all three public paths.
The `/pe-quant` deny rule is retained unchanged.

## Build and verify

```bash
cd /home/ec2-user/exp/apps/connectome_web
npm ci
npm run generate:api-types
npm audit --audit-level=moderate
npm run typecheck
npm run build
npm run test:e2e

cd /home/ec2-user/exp
.venv_connectome_api/bin/python -m pytest tests/dashboard_parity -q
curl -fsS http://127.0.0.1:8001/api/v1/health/ready
```

Current accepted shadow baseline (2026-07-29):

- Python core/API/reference/security suite: 23 passed;
- real-Chromium Playwright suite: 7 passed;
- TypeScript typecheck and production build: passed;
- `npm audit --audit-level=high`: zero vulnerabilities;
- bounded health, network-distribution and pipeline load checks: zero failed
  requests;
- stop/restore rollback rehearsal: passed while Streamlit remained healthy.

The detailed evidence and frozen bundle hashes are recorded in
`research_audit/FASTAPI_REACT_MIGRATION_VALIDATION_20260729.md`.

## Service operations

```bash
sudo systemctl status connectome-api.service
sudo journalctl -u connectome-api.service -n 100 --no-pager
sudo systemctl restart connectome-api.service
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:8001/api/v1/health/live
curl -fsS http://127.0.0.1:8001/api/v1/health/ready
curl -fsS http://127.0.0.1:8001/next/
```

## Safe rollback

The Streamlit root is not replaced during shadow migration. The rehearsed
shadow-only rollback is:

1. Restore the dated nginx backup created at deployment.
2. Run `sudo nginx -t` and `sudo systemctl reload nginx`.
3. Stop and disable `connectome-api.service`.
4. Confirm `http://127.0.0.1:8501/_stcore/health` remains 200 and the public
   root still resolves to Streamlit through the authenticated nginx boundary.

No scientific artifact, Streamlit source, production worker, or pipeline
ledger is modified by this procedure.

## Root-promotion gate

Do not promote React from `/next/` to `/` until both conditions are explicit:

1. the user completes visual and keyboard acceptance of the protected shadow;
2. the user explicitly authorizes the public-root change.

Before promotion, save a dated copy of the active nginx configuration. After
the approved edit:

1. run `sudo nginx -t`;
2. reload nginx without stopping Streamlit or the HCP workers;
3. verify authenticated `/`, `/api/v1/health/ready`, all primary routes and
   downloads;
4. verify unauthenticated `/` and `/api/` remain 401;
5. retain Streamlit on `127.0.0.1:8501` as the immediate rollback target until
   post-promotion acceptance is complete.

The root has not been promoted as of the accepted shadow baseline.
