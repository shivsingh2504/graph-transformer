# Graph Transformer demo

The Streamlit site introduces weighted shortest paths with a small worked example, diagrams the encoder-decoder flow, and presents Run 8 achievements before the interactive Graph Lab. The graph view highlights node roles and predicted links, with edge costs legible at normal viewing size. It uses a navy, teal, blue, and amber palette. The model's supported input range is 5–20 nodes; Run 8 reached 91.1% valid-and-optimal paths across 1,000 in-distribution examples and passed the 90% gate.

## Use the demo locally

### Docker

Install Docker Desktop, then run from the repository root:

```sh
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The API is also available at `http://localhost:8000/api`. Stop the services with `docker compose down`.

### Python

Install dependencies from `requirements_clean.txt`, then start the API in one terminal and Streamlit in another:

```sh
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

Set `API_BASE_URL` when the API is not at `http://127.0.0.1:8000/api`.

## Use the graph lab

Choose a graph size from 5 through 20 and an optional whole-number seed, generate a graph, select different start and target nodes, then run inference. Green indicates a valid optimal path; yellow indicates a valid but longer path; red indicates an invalid prediction. Dijkstra’s path is shown as the exact reference. The graph is keyboard-focusable; all controls are native Streamlit widgets with visible labels and help text.

## Hosting and CI

`render.yaml` describes one free public Streamlit web service on Render. It runs the finalized FastAPI model API on loopback inside the same container, so the API itself is not exposed to the public internet. The GitHub Actions workflow builds the container on pushes to `main` and `run8`, and on pull requests. Render can deploy the Blueprint automatically after it is connected to the repository. See the [Render Blueprint reference](https://render.com/docs/blueprint-spec) for service configuration.

### Private checkpoint setup for Render

The verified `checkpoints_run8/final.pt` is intentionally excluded from Git and Docker build contexts. At startup, the web service downloads it from the private HTTPS artifact URL in `CHECKPOINT_URL` and verifies its state-dictionary SHA-256 against the verified Run 8 hash. The finalized `src/api.py` remains unchanged.

The no-cost Render setup has free-tier limits: the site can sleep after 15 minutes without traffic and take about a minute to wake, and the free filesystem is erased on sleep, restart, and redeploy. The checkpoint therefore downloads again after each wake or restart, and the private URL must still be valid then. Free services also use the workspace's monthly instance-hour, bandwidth, and build quotas; Render may suspend service or builds when included quotas are exhausted. See [Render's free service limits](https://render.com/docs/free).

Before the first Render deploy:

1. Upload `checkpoints_run8/final.pt` to private object storage that can provide a direct HTTPS download URL. Keep the object private; use a signed URL or equivalent read-only access.
2. Create the Render Blueprint from `render.yaml` using the `run8` branch.
3. When Render prompts for the unsynchronized `CHECKPOINT_URL` value, paste the private download URL directly into Render. The bootstrap never prints the URL or stores it in the image.

Do not commit the checkpoint or its signed URL to Git. If the URL expires, update the `CHECKPOINT_URL` secret in Render before the next restart or wake.

## Project notes

- `ui/app.py` contains the Streamlit interface.
- `src/api.py` remains the finalized inference API and loads `checkpoints_run8/final.pt`.
- The evaluation summary is JSON stored in `checkpoints_run8/results.txt` and shown in the results panel.
- The Docker image contains both API and UI entry points; Compose runs them as separate services.
