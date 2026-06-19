"""
celery_config.py
================
Celery application factory for MINI-TOURISM RAG.

Broker  : RabbitMQ  (AMQP)
Backend : Redis     (result store + optional cache)

Queues
------
  processing   – CPU-bound document chunking tasks
  indexing     – Network-bound embedding + vector-DB push tasks
  default      – Catch-all / lightweight tasks

Usage
-----
  # In any Python module:
  from celery_app.celery_config import celery_app
"""

from __future__ import annotations

import os
import logging
from celery import Celery
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

# ── Connection URLs (read from env with sane defaults) ──────────────────────

def _broker_url() -> str:
    user     = os.getenv("RABBITMQ_DEFAULT_USER", "minirag")
    password = os.getenv("RABBITMQ_DEFAULT_PASS", "minirag_rabbit_2222")
    host     = os.getenv("RABBITMQ_HOST", "rabbitmq")
    port     = os.getenv("RABBITMQ_PORT", "5672")
    vhost    = os.getenv("RABBITMQ_VHOST", "minirag_vhost")
    return f"amqp://{user}:{password}@{host}:{port}/{vhost}"


def _result_backend() -> str:
    password = os.getenv("REDIS_PASSWORD", "minirag_redis_2222")
    host     = os.getenv("REDIS_HOST", "redis")
    port     = os.getenv("REDIS_PORT", "6379")
    db       = os.getenv("REDIS_CELERY_DB", "1")       # DB 1 — keep 0 for cache
    return f"redis://:{password}@{host}:{port}/{db}"


# ── Exchange & Queue definitions ────────────────────────────────────────────

_default_exchange   = Exchange("default",    type="direct", durable=True)
_processing_exchange = Exchange("processing", type="direct", durable=True)
_indexing_exchange   = Exchange("indexing",   type="direct", durable=True)

TASK_QUEUES = (
    Queue("default",    _default_exchange,    routing_key="default"),
    Queue("processing", _processing_exchange, routing_key="processing"),
    Queue("indexing",   _indexing_exchange,   routing_key="indexing"),
)

# ── Task routing rules ───────────────────────────────────────────────────────

TASK_ROUTES = {
    "celery_app.tasks.processing.*": {
        "queue": "processing",
        "routing_key": "processing",
    },
    "celery_app.tasks.indexing.*": {
        "queue": "indexing",
        "routing_key": "indexing",
    },
}


# ── Application factory ─────────────────────────────────────────────────────

def create_celery_app() -> Celery:
    app = Celery(
        "minirag",
        broker=_broker_url(),
        backend=_result_backend(),
        include=[
            "celery_app.tasks.processing",
            "celery_app.tasks.indexing",
        ],
    )

    app.conf.update(
        # ── Serialisation ────────────────────────────────────────────────────
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # ── Queues ───────────────────────────────────────────────────────────
        task_queues=TASK_QUEUES,
        task_default_queue="default",
        task_default_exchange="default",
        task_default_routing_key="default",
        task_routes=TASK_ROUTES,
        # ── Worker behaviour ─────────────────────────────────────────────────
        worker_prefetch_multiplier=1,       # Fair task distribution
        task_acks_late=True,                # Re-queue on worker crash
        task_reject_on_worker_lost=True,
        # ── Result TTL (24 h) ────────────────────────────────────────────────
        result_expires=86_400,
        # ── Retry / rate limits ──────────────────────────────────────────────
        task_max_retries=3,
        task_default_retry_delay=5,         # seconds
        # ── Timezone ────────────────────────────────────────────────────────
        timezone="UTC",
        enable_utc=True,
        # ── Beat schedule (optional periodic tasks) ──────────────────────────
        beat_schedule={},
    )

    logger.info(
        "Celery app created | broker=%s | backend=%s",
        _broker_url().split("@")[-1],   # hide credentials
        _result_backend().split("@")[-1],
    )
    return app


# Module-level singleton — import this everywhere
celery_app: Celery = create_celery_app()
