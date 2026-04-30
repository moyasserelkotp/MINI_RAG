#!/bin/bash
set -e

# Note: Alembic migrations skipped - using MongoDB instead of SQL database
# If you need database migrations, uncomment the lines below and ensure alembic is in requirements.txt
# echo "Running database migrations..."
# cd /app/models/db_schemes/minirag/
# alembic upgrade head
# cd /app

echo "Starting FastAPI application..."
exec "$@"