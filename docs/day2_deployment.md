# Day 2 Deployment

## URLs
- Frontend: https://smart-learn-ai-weld.vercel.app
- Backend health: https://smartlearn-ai-production-2c4e.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-2c4e.up.railway.app/docs

## Source
- Repository: student's fork (HOT-KNIFE/smartLearn-AI)
- Deployed branch / merge target: main
- Merged commit: 7822dd6
- Pull Request: https://github.com/HOT-KNIFE/smartLearn-AI/pull/1

## Root Directories
- Railway: smartlearn-backend
- Vercel: smartlearn-frontend

## Environment variable names
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass
- Upload: pass
- Known /chat + citations: pass
- Unknown question: pass
- CORS restart + re-upload recovery: pass

## Known limitations
- Railway restart clears in-memory uploaded/chat state; re-upload is expected.
