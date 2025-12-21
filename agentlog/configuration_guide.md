# Configuration Guide

This system relies on environment variables for configuration. Copy `config.env.example` to `config.env` and update the values.

## Core Configuration

| Variable | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `GOOGLE_API_KEY` | **Yes** | API Key for Gemini/Gemma models. | `AIzaSy...` |
| `SUPABASE_URL` | **Yes** | URL of your Supabase project. | `https://xyz.supabase.co` |
| `SUPABASE_KEY` | **Yes** | Anon/Public key for Supabase. | `eyJ...` |
| `HF_TOKEN` | No | HuggingFace token for downloading embedding models. | `hf_...` |

## OCR Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_LOCAL_MARKER` | `true` | Set to `true` to use local `marker-pdf` (free). Set to `false` for Datalab API. |
| `DATALAB_API_KEY` | - | Required if `USE_LOCAL_MARKER=false`. API key for Datalab. |
| `DATALAB_API_URL` | `https://www.datalab.to/api/v1/marker` | API endpoint for Datalab. |

## Development Mode

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEV_MODE` | `false` | If `true`, bypasses JWT authentication and uses a mock user. **Do not use in production.** |

## Model Configuration (Internal Defaults)

These are configured in `core/llm_factory.py` and are not currently exposed as env vars, but good to know:

*   **Translation Model**: `gemini-3.0-flash`
*   **Graph Extraction**: `gemini-3.0-flash`
*   **Community Summary**: `gemini-3.0-flash`
*   **General Model**: `gemma-3-27b-it`
