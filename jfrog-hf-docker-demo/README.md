# JFrog Artifactory + Hugging Face Docker Demo

Docker image with a **Hugging Face model baked in** (no download at runtime), suitable for pushing to a JFrog Artifactory Docker repository.

## What’s included

- **FastAPI app** that loads the model from `/app/model` at startup and exposes:
  - `GET /health` – health check
  - `POST /predict` – text-to-text generation (body: `{"text": "Your prompt or question"}`)
  - `GET /docs` – Swagger UI
- **Dockerfile** that downloads the Hugging Face model at **build time** and bakes it into the image (default: **google/flan-t5-small** — small, popular FLAN-T5 for summarization, Q&A, etc.).
- **Script** to build and push the image to Artifactory.

## Install Docker (macOS)

If Docker is not installed, run in **Terminal** (you’ll be prompted for your password):

```bash
cd jfrog-hf-docker-demo
./install-docker-mac.sh
```

This installs Homebrew (if needed) and Docker Desktop. Then open **Docker Desktop** from Applications, accept the terms, and wait until it’s running before using `docker` commands.

## Build the image locally

```bash
cd jfrog-hf-docker-demo
docker build -t jfrog-hf-demo:latest .
```

Optional: use a different Hugging Face model (e.g. sentiment):

```bash
docker build --build-arg HF_MODEL_ID=distilbert-base-uncased-finetuned-sst-2-english -t jfrog-hf-demo:latest .
```
Note: switching the model also requires matching app code (e.g. sentiment vs text2text).

## Run locally

```bash
docker run -p 8000:8000 jfrog-hf-demo:latest
```

Then:

- Open http://localhost:8000/docs
- Try `POST /predict` with body: `{"text": "Summarize: Machine learning is a subset of artificial intelligence."}` or `{"text": "Question: What is the capital of France? Answer:"}`

## Push to JFrog Artifactory

1. Set your Artifactory Docker repo and (optionally) credentials:

   ```bash
   export ARTIFACTORY_URL="your-instance.jfrog.io"
   export ARTIFACTORY_REPO="docker-local"   # your Docker repo key
   export ARTIFACTORY_USER="your-username"
   export ARTIFACTORY_PASSWORD="your-password-or-api-key"
   ```

2. Run the script:

   ```bash
   chmod +x push_to_jfrog.sh
   ./push_to_jfrog.sh
   ```

   This builds the image, tags it as `{ARTIFACTORY_URL}/{ARTIFACTORY_REPO}/jfrog-hf-demo:latest`, logs in to Artifactory (if user/password are set), and pushes.

3. Optional env vars:
   - `DOCKER_IMAGE_NAME` – image name (default: `jfrog-hf-demo`)
   - `IMAGE_TAG` – tag (default: `latest`)
   - `HF_MODEL_ID` – Hugging Face model ID (default: `google/flan-t5-small`)

## Pull and run from Artifactory

After pushing:

```bash
docker login your-instance.jfrog.io -u YOUR_USER -p YOUR_PASSWORD
docker pull your-instance.jfrog.io/docker-local/jfrog-hf-demo:latest
docker run -p 8000:8000 your-instance.jfrog.io/docker-local/jfrog-hf-demo:latest
```
