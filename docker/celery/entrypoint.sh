#!/bin/bash
# ==============================================================================
# Celery worker entrypoint
# Waits for RabbitMQ and Redis to be available before starting.
# ==============================================================================
set -e

# ── Wait for RabbitMQ ─────────────────────────────────────────────────────────
RABBIT_HOST="${RABBITMQ_HOST:-rabbitmq}"
RABBIT_PORT="${RABBITMQ_PORT:-5672}"

echo "[worker] Waiting for RabbitMQ at ${RABBIT_HOST}:${RABBIT_PORT}..."
until nc -z "${RABBIT_HOST}" "${RABBIT_PORT}" 2>/dev/null; do
  echo "[worker] RabbitMQ not ready — sleeping 2s..."
  sleep 2
done
echo "[worker] RabbitMQ is up."

# ── Wait for Redis ────────────────────────────────────────────────────────────
REDIS_HOST_VAR="${REDIS_HOST:-redis}"
REDIS_PORT_VAR="${REDIS_PORT:-6379}"

echo "[worker] Waiting for Redis at ${REDIS_HOST_VAR}:${REDIS_PORT_VAR}..."
until nc -z "${REDIS_HOST_VAR}" "${REDIS_PORT_VAR}" 2>/dev/null; do
  echo "[worker] Redis not ready — sleeping 2s..."
  sleep 2
done
echo "[worker] Redis is up."

echo "[worker] Starting Celery worker..."
exec "$@"
