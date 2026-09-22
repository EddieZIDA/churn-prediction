# Deployment Guide

Ce guide explique comment déployer l'application Churn Prediction en production.

---

## 🚀 Streamlit Cloud (Recommandé - Gratuit)

### Prérequis

1. Compte GitHub avec le repo `churn-prediction`
2. Compte Streamlit Cloud (gratuit via `streamlit.io/cloud`)
3. Repo public OU invite collaborateur dans repo privé

### Étapes de Déploiement

#### 1. Connecter le Repo

```bash
1. Allez sur https://streamlit.io/cloud
2. Cliquez "New app"
3. Connectez votre GitHub (authorize si demandé)
4. Sélectionnez le repo `churn-prediction`
5. Branch: `main`
6. App path: `streamlit_app.py`
7. Cliquez "Deploy"
```

#### 2. Configurer Variables d'Environnement

Dans Streamlit Cloud dashboard:

```
Settings → Secrets
Ajouter:
  LOG_LEVEL=INFO
  LOG_FILE=/app/logs/app.log
  MLFLOW_MODEL_URI=(optionnel, si le modèle vient d'un registry MLflow)
```

Ce sont les seules variables lues par le code (`src/logger.py`,
`src/model.py`). Le chemin du modèle n'est pas configurable par variable
d'environnement : il est dérivé de `src/config.py`.

#### 3. Vérifier le Déploiement

```
- App URL sera: https://churn-prediction-{random}.streamlit.app/
- Status: "Your app is live!"
- Ouvrez l'URL et testez l'app
```

### Auto-Redeploy

✅ Automatique à chaque push sur `main`!

- Push code → GitHub Actions tests run
- Streamlit Cloud détecte le push et reconstruit l'application
- Les règles de protection GitHub garantissent que les PR passent les checks avant merge
- Logs visibles en "Manage app" → "View logs"

### Monitoring

```
Streamlit Cloud Dashboard:
├─ Health status (green = live)
├─ App URL
├─ Recent deployments
├─ View logs (real-time)
└─ Settings (secrets, resources)
```

### Troubleshooting

**App crashes?**
```
1. Check logs: Dashboard → View logs
2. Check requirements.txt is complete
3. Verify MODEL_PATH exists in repo
4. Re-deploy manually if needed
```

**Slow startup?**
```
1. Streamlit Cloud rebuilds every deploy (~2-3 min)
2. Model loading is cached (@st.cache_resource)
3. First visit slower, then cached
```

**Can't connect to model?**
```
1. Verify models/best_model.pkl exists in repo
2. Check MODEL_PATH in src/config.py
3. Verify model file size < 500MB
4. If model > 500MB, use MLflow or cloud storage
```

---

## 🐳 Docker & Docker Hub (Avancé)

### Build Local Docker Image

```bash
docker build -t churn-prediction:v1.0 .
```

### Run Local Container

```bash
docker run -p 8501:8501 churn-prediction:v1.0
# Ouvrez http://localhost:8501
```

### Push to Docker Hub

```bash
# 1. Create repo on hub.docker.com
# 2. Login
docker login
docker tag churn-prediction:v1.0 {username}/churn-prediction:v1.0

# 3. Push
docker push {username}/churn-prediction:v1.0

# 4. Public URL: https://hub.docker.com/r/{username}/churn-prediction
```

### Deploy Docker Container on VPS

```bash
# SSH to VPS
ssh user@vps-ip

# Pull image
docker pull {username}/churn-prediction:v1.0

# Run container
docker run -d \
  -p 80:8501 \
  --name churn-app \
  --restart always \
  {username}/churn-prediction:v1.0

# App accessible: http://vps-ip/
```

---

## 🔄 CI/CD Workflow

### Automatisé à Chaque Commit

```
Developer pushes code
       ↓
GitHub Actions triggers
       ├─ Job 1: Lint check (black, flake8)
       ├─ Job 2: Run tests (pytest)
       ├─ Job 3: Check coverage (80%)
       └─ Job 4: Security check (bandit)
       ↓
All jobs pass?
       ├─ YES → Merge to main
       │         ↓
       │         Auto-deploy to Streamlit Cloud
       │         ↓
       │         App updated live (~3 min)
       │
       └─ NO → PR review fails
                Must fix & push again
```

### Redéploiement manuel

Le workflow `deploy.yml` se déclenche uniquement à la suite du workflow de
tests (`workflow_run`) : il n'a pas de déclencheur manuel. Pour redéployer
sans changement de code :

```
Streamlit Cloud → Manage app → Reboot app
```

---

## 📊 Monitoring & Logs

### Streamlit Cloud Logs

```
Dashboard → Manage app → View logs

Example log output:
  2024-09-01 10:30:45,123 [INFO] [model.py] Model loaded successfully
  2024-09-01 10:30:46,456 [INFO] [streamlit_app.py] Prediction made: 0.72
  2024-09-01 10:30:47,789 [DEBUG] [explainer.py] SHAP calculated
```

### Local Testing

```bash
# Test with same config as production
LOG_LEVEL=INFO streamlit run streamlit_app.py
```

### Check Deployed Logs

```
Streamlit Cloud → Manage app → View logs
(Near real-time, last 24h)
```

---

## 🔐 Secrets Management

### Sensitive Data (Passwords, Keys)

**❌ NEVER commit to GitHub:**
- Passwords
- API keys
- Database credentials
- Model URIs with auth

**✅ Use Streamlit Cloud Secrets:**

```
Settings → Secrets → Paste:
  MLFLOW_TRACKING_URI=https://...
  MLFLOW_TRACKING_USERNAME=user
  MLFLOW_TRACKING_PASSWORD=secret
```

In code:
```python
import streamlit as st

mlflow_uri = st.secrets["MLFLOW_TRACKING_URI"]
mlflow_user = st.secrets["MLFLOW_TRACKING_USERNAME"]
```

---

## 🚨 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| App won't load | Missing dependency | Add to requirements.txt |
| Slow startup | Large model file | Use cache decorator |
| Tests fail in CI | Local vs CI environment | Check Python version match |
| Deploy fails | Secrets not configured | Set in Streamlit Cloud |
| Model not found | Path issue | Verify MODEL_PATH relative |
| Memory issues | Large dataset | Reduce test size or batch |

---

## ✅ Deployment Checklist

Before pushing to main:

- [ ] All tests pass locally: `pytest tests/ -q`
- [ ] Coverage 80%+: `pytest tests/ --cov=src`
- [ ] Linting passes: `flake8 src/ tests/ streamlit_app.py`
- [ ] Format check: `black --check src/ tests/ streamlit_app.py`
- [ ] No hardcoded secrets in code
- [ ] requirements.txt updated
- [ ] README / REPORT updated

Push to main (or merge PR):

- [ ] GitHub Actions all pass
- [ ] Streamlit Cloud deploys (check logs)
- [ ] Live app loads: https://churn-prediction-xxx.streamlit.app/
- [ ] Basic functionality works
- [ ] Test with production data

---

## 📞 Support & Escalation

| Problem | Who | Action |
|---------|-----|--------|
| App down | Ops team | Check Streamlit Cloud dashboard |
| Slow response | Dev team | Check logs, optimize model |
| Data issue | Data team | Verify data pipeline |
| Security incident | Security team | Review logs, update secrets |

---

**Pour aller plus loin** : [REPORT.md](REPORT.md) — architecture, chaîne de
qualité (CI), et guide de contribution.

