# Image de production de l'API de scoring (Hugging Face Spaces, ou tout hôte Docker)
FROM python:3.13.12-slim

# LightGBM a besoin de la bibliothèque OpenMP, absente de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv, à la même version que sur le poste de développement
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /bin/uv

# Hugging Face Spaces exécute le conteneur avec l'utilisateur 1000 : on ne tourne pas en root
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/home/user/app/.venv/bin:$PATH"

# Dépendances d'abord : cette couche reste en cache tant que uv.lock ne change pas
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# Puis le code et le modèle, qui changent plus souvent
COPY --chown=user api/ api/
COPY --chown=user model/ model/

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
