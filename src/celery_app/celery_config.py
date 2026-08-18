from __future__ import annotations

import os
import logging
from celery import Celery
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

#  Connection URLs (read from env with sane defaults) 
def _broker_url() -> str:
    try:
        user     = os.environ["RABBITMQ_DEFAULT_USER"]
        password = os.environ["RABBITMQ_DEFAULT_PASS"]
        host     = os.getenv("RABBITMQ_HOST", "rabbitmq")
        port     = os.getenv("RABBITMQ_PORT", "5672")
        vhost    = os.getenv("RABBITMQ_VHOST", "minirag_vhost")
        return f"amqp://{user}:{password}@{host}:{port}/{vhost}"
    except KeyError as e:
        raise ValueError(f"Missing required RabbitMQ environment variable for Celery: {e}")


def _result_backend() -> str:
    try:
        password = os.environ["REDIS_PASSWORD"]
        host     = os.getenv("REDIS_HOST", "redis")
        port     = os.getenv("REDIS_PORT", "6379")
        db       = os.getenv("REDIS_CELERY_DB", "1")       
        return f"redis://:{password}@{host}:{port}/{db}"
    except KeyError as e:
        raise ValueError(f"Missing required Redis environment variable for Celery: {e}")


#  Exchange & Queue definitions 
_default_exchange   = Exchange("default",    type="direct", durable=True)
_processing_exchange = Exchange("processing", type="direct", durable=True)
_indexing_exchange   = Exchange("indexing",   type="direct", durable=True)
_dlx_exchange        = Exchange("dlx",        type="direct", durable=True)

TASK_QUEUES = (
    Queue("default",    _default_exchange,    routing_key="default",
        queue_arguments={"x-dead-letter-exchange": "dlx", "x-dead-letter-routing-key": "dead_letter"}),
    Queue("processing", _processing_exchange, routing_key="processing",
        queue_arguments={"x-dead-letter-exchange": "dlx", "x-dead-letter-routing-key": "dead_letter"}),
    Queue("indexing",   _indexing_exchange,   routing_key="indexing",
        queue_arguments={"x-dead-letter-exchange": "dlx", "x-dead-letter-routing-key": "dead_letter"}),
    Queue("dead_letters", _dlx_exchange,      routing_key="dead_letter"),
)

#  Task routing rules 
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


#  Application factory 
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
        #  Serialisation 
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        #  Queues 
        task_queues=TASK_QUEUES,
        task_default_queue="default",
        task_default_exchange="default",
        task_default_routing_key="default",
        task_routes=TASK_ROUTES,
        #  Worker behaviour 
        worker_prefetch_multiplier=1,       # Fair task distribution
        task_acks_late=True,                # Re-queue on worker crash
        task_reject_on_worker_lost=True,
        #  Result TTL (24 h) 
        result_expires=86_400,
        #  Retry / rate limits 
        task_max_retries=3,
        task_default_retry_delay=5,         # seconds
        #  Timezone 
        timezone="UTC",
        enable_utc=True,
        #  Startup connection retry (required for Celery 6.0+ compatibility)
        broker_connection_retry_on_startup=True,
        #  Beat schedule (optional periodic tasks) 
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


# CEL-04: Dead-letter queue monitor task
# ----------------------------------------
# Call this periodically via Celery Beat (uncomment beat_schedule entry below)
# or trigger ad-hoc: celery_app.send_task("celery_app.celery_config.monitor_dead_letters")
@celery_app.task(name="celery_app.celery_config.monitor_dead_letters", queue="default")
def monitor_dead_letters():
    """Log the number of messages sitting in the dead_letters queue.

    Wire into beat_schedule to run on a schedule, e.g.::

        beat_schedule = {
            "dead-letter-monitor": {
                "task": "celery_app.celery_config.monitor_dead_letters",
                "schedule": 300,  # every 5 minutes
            }
        }
    """
    try:
        with celery_app.connection_for_read() as conn:
            with conn.channel() as ch:
                _, msg_count, _ = ch.queue_declare("dead_letters", passive=True)
        if msg_count:
            logger.warning(
                "CEL-04 | dead_letters queue has %d unprocessed message(s). "
                "Inspect via RabbitMQ management UI or re-queue manually.",
                msg_count,
            )
        else:
            logger.info("CEL-04 | dead_letters queue is empty.")
        return {"dead_letter_count": msg_count}
    except Exception as exc:
        logger.error("CEL-04 | Failed to inspect dead_letters queue: %s", exc)
        return {"error": str(exc)}


"""
celery_config.py
================
Celery application factory for MINI-RAG.

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