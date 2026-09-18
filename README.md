# US Data center buildout - A Tilebox Workflow

This repository contains the Tilebox workflow behind the [US data center growth tracker](https://datacenterbuildout.com/).

The tracker compares satellite imagery around known data center sites, ranks where visible construction change happened, and publishes the evidence needed for a browsable data product: before/after previews, change scores, scene metadata, and a final ranking JSON.

It was built with an agent on Tilebox. The point of the demo is simple: do not just ask an agent a geospatial question and accept a static answer. Give the agent data, compute, workflow code, logs, artifacts, and observability so it can build a product you can inspect and rerun.

- Result: <https://datacenterbuildout.com/>
- Original agent transcript: <https://ampcode.com/threads/T-019eaba7-ef4b-718e-b850-97f5df3baf0f>

## What the workflow does

The root task is:

```text
tilebox.com/datacenters/RankDataCenterBuildout@v1.14
```

For each candidate site, it:

1. loads a CSV of known or proposed data center locations
2. filters and merges nearby duplicate points
3. finds a clear Sentinel-2 L2A scene before and after the target dates
4. crops the imagery around the site
5. computes construction-oriented change signals
6. compares the before/after imagery with Clay foundation model embeddings
7. writes ranked results to `outputs/ranking.json` in the Tilebox job cache

You do not need to be a geospatial expert to start. Think of the workflow as: “take a list of places, get satellite images before and after, score which places changed most, and save evidence for review.”

## Useful inputs

The root task accepts these common parameters:

```json
{
  "csv_url": "https://docs.google.com/spreadsheets/d/1JJ6kcVo-NjlAYtznwHOki2DVl4WWV6lhy-eXhFCdKKU/export?format=csv&gid=386766486",
  "max_sites": 3,
  "random_seed": 1337,
  "before_date": "2024-05-01",
  "after_date": "2026-05-01",
  "window_days": 60,
  "crop_size_m": 3000,
  "scene_cloud_cover_max": 30.0,
  "crop_cloud_cover_max": 1.0,
  "status_filter": [
    "Approved/Permitted/Under construction",
    "Expanding",
    "Proposed"
  ]
}
```

Notes:

- `max_sites` is the easiest way to keep early runs cheap and fast.
- `before_date` and `after_date` define the comparison period.
- `window_days` lets the workflow search around those dates for usable low-cloud imagery.
- `crop_size_m` controls how much area around each site is analyzed.
- If `status_filter` is omitted, the workflow defaults to approved, expanding, and proposed sites.

## Adapting it with an agent

This project is meant to be changed by coding agents. Good agent instructions are product-oriented and include how to verify the result.

Try prompts like:

```text
Read this repository and explain the workflow in plain English. Then run a 3-site smoke test, inspect the Tilebox job, and summarize whether the outputs are usable.
```

```text
Publish and deploy this workflow to my Tilebox cluster. Submit a small job with max_sites=5, wait for it to finish, inspect failures or low-quality results, and make the smallest code changes needed.
```

```text
Adapt this data center workflow to rank visible construction at solar farm sites. Replace the input CSV schema as needed, keep the Sentinel-2 before/after scene selection, and adjust the scoring so large new bright panel-like areas rank higher.
```

```text
Make this workflow easier to use for non-geospatial users. Add clearer task display names, better log messages, and output fields that explain why a site ranked highly.
```

```text
Use the latest completed job to build a small static website from outputs/ranking.json and the cached preview images. Keep the page simple: map, ranked list, before/after evidence, and score details.
```

```text
Run the workflow on 30 sampled sites, compare the top-ranked results manually from the previews, and propose scoring changes to reduce vegetation-only false positives.
```

## What to take away

Data centers are just one example. The same pattern works for ports, agriculture, mining, energy, disaster recovery, parking lots, supply chains, or any question that depends on how places change over time.

Tilebox gives the agent the loop it needs to build something real:

```text
prompt → workflow code → deployed compute → observable job → inspectable outputs → iteration → data product
```

Clone the repo, run a small version, deploy it to a runner, or point your own agent at it and adapt it to a different idea.

Do not just ask a question — build the product.


## Requirements

- Python 3.12 and `uv`
- Tilebox CLI, and a Tilebox API key from the [Tilebox Console](https://console.tilebox.com) - sign up for free!
- [Copernicus Data Space S3 credentials](https://docs.tilebox.com/storage/clients#access-copernicus-data)

```bash
export COPERNICUS_ACCESS_KEY="..."
export COPERNICUS_SECRET_KEY="..."
```

The workflow lazily downloads the Clay v1.5 checkpoint on first use and caches it under `~/.cache/tilebox/models/`. 

## Run a small job

Install dependencies:

```bash
uv sync
```

For local development without Google Cloud credentials, use the local workflow cache:

```bash
export WORKFLOW_CACHE_BUCKET=""
```

Publish and deploy the workflow release:

```bash
RELEASE_ID=$(tilebox workflow publish-release --json | jq -r '.id')
tilebox workflow deploy-release --release "$RELEASE_ID" --cluster "<your-cluster>" --json
```

Submit a small smoke test first:

```bash
tilebox job submit \
  --name datacenter-buildout-smoke \
  --task tilebox.com/datacenters/RankDataCenterBuildout \
  --version v1.14 \
  --cluster "<your-cluster>" \
  --input '{
    "max_sites": 3,
    "random_seed": 1337,
    "before_date": "2024-05-01",
    "after_date": "2026-05-01",
    "window_days": 60,
    "crop_size_m": 3000,
    "scene_cloud_cover_max": 30.0,
    "crop_cloud_cover_max": 1.0
  }' \
  --wait \
  --json
```

After the job runs, inspect it in the Tilebox Console. Look at task logs, spans, inputs, cached previews, scene metadata, and `outputs/ranking.json`. That inspection loop is the important part: the agent can use the same evidence to debug and improve the workflow.
