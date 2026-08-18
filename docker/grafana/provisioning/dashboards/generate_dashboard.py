import json

markdown_content = """# MINI-TOURISM RAG
This project provides a robust Retrieval-Augmented Generation (RAG) backend tailored for internal tourism knowledge and documentation.

### Components
- **FastAPI**: Serves the REST API, coordinates LLMs (OpenAI/Cohere), and orchestrates data pipelines.
- **MongoDB**: Stores Projects, Sessions, Messages, Assets, and Document Chunks.
- **Qdrant**: High-performance Vector Database for semantic search.
- **Celery & Redis**: Handles async document processing and indexing.
- **Prometheus & Grafana**: Collects and visualizes application metrics.

### Key Metrics Monitored
- **HTTP**: `http_requests_total`, `http_request_duration_seconds`
- **Retrieval & Cache**: `rag_cache_hits_total`, `rag_cache_misses_total`, `rag_retrieval_duration_seconds`, `rag_chunks_retrieved_total`, `rag_retrieval_errors_total`
- **Generation**: `rag_generation_tokens_total`, `rag_generation_duration_seconds`, `rag_generation_errors_total`
- **Processing**: `rag_documents_processed_total`, `rag_chunking_duration_seconds`
"""

dashboard = {
    "title": "FastAPI RAG Metrics",
    "timezone": "browser",
    "schemaVersion": 38,
    "refresh": "10s",
    "panels": [
        {
            "type": "text",
            "title": "MINI-TOURISM RAG - Project Overview",
            "gridPos": {"h": 7, "w": 24, "x": 0, "y": 0},
            "options": {
                "mode": "markdown",
                "content": markdown_content
            }
        },
        # HTTP Metrics Row
        {
            "type": "timeseries",
            "title": "HTTP Request Rate",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 7},
            "targets": [
                {
                    "expr": "sum(rate(http_requests_total[5m])) by (endpoint, status)",
                    "legendFormat": "{{endpoint}} - {{status}}"
                }
            ]
        },
        {
            "type": "timeseries",
            "title": "HTTP Latency (p95)",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 7},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))",
                    "legendFormat": "{{endpoint}}"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "s"}}
        },
        # Retrieval & Cache Row
        {
            "type": "timeseries",
            "title": "RAG Cache Hits vs Misses",
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 15},
            "targets": [
                {
                    "expr": "sum(rate(rag_cache_hits_total[5m])) by (project_id)",
                    "legendFormat": "Hits - {{project_id}}"
                },
                {
                    "expr": "sum(rate(rag_cache_misses_total[5m])) by (project_id)",
                    "legendFormat": "Misses - {{project_id}}"
                }
            ]
        },
        {
            "type": "timeseries",
            "title": "Retrieval Latency (p95)",
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 15},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(rag_retrieval_duration_seconds_bucket[5m])) by (le, project_id))",
                    "legendFormat": "{{project_id}}"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "s"}}
        },
        {
            "type": "timeseries",
            "title": "Avg Chunks Retrieved per Query",
            "gridPos": {"h": 8, "w": 6, "x": 12, "y": 15},
            "targets": [
                {
                    "expr": "sum(rate(rag_chunks_retrieved_total_sum[5m])) / sum(rate(rag_chunks_retrieved_total_count[5m])) > 0",
                    "legendFormat": "Chunks"
                }
            ]
        },
        {
            "type": "timeseries",
            "title": "Retrieval Errors",
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 15},
            "targets": [
                {
                    "expr": "sum(rate(rag_retrieval_errors_total[5m])) by (project_id)",
                    "legendFormat": "{{project_id}}"
                }
            ]
        },
        # LLM Generation Row
        {
            "type": "timeseries",
            "title": "LLM Generation Tokens / sec",
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 23},
            "targets": [
                {
                    "expr": "sum(rate(rag_generation_tokens_total[5m])) by (backend)",
                    "legendFormat": "{{backend}}"
                }
            ]
        },
        {
            "type": "timeseries",
            "title": "Generation Latency (p95)",
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 23},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(rag_generation_duration_seconds_bucket[5m])) by (le, backend))",
                    "legendFormat": "{{backend}}"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "s"}}
        },
        {
            "type": "timeseries",
            "title": "Generation Errors",
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 23},
            "targets": [
                {
                    "expr": "sum(rate(rag_generation_errors_total[5m])) by (backend)",
                    "legendFormat": "{{backend}}"
                }
            ]
        },
        # Data Processing Row
        {
            "type": "timeseries",
            "title": "Documents Processed",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 31},
            "targets": [
                {
                    "expr": "sum(rate(rag_documents_processed_total[5m])) by (project_id)",
                    "legendFormat": "{{project_id}}"
                }
            ]
        },
        {
            "type": "timeseries",
            "title": "Chunking Latency (p95)",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 31},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(rag_chunking_duration_seconds_bucket[5m])) by (le, strategy))",
                    "legendFormat": "{{strategy}}"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "s"}}
        }
    ]
}

with open("d:/python/R_A_G/MINI-TOURISM_RAG/docker/grafana/provisioning/dashboards/fastapi-rag.json", "w") as f:
    json.dump(dashboard, f, indent=2)

print("Dashboard successfully generated.")
