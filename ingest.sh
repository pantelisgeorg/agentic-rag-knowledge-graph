#!/usr/bin/env bash
# Ingest documents into the vector DB + knowledge graph.
# Prerequisites: PostgreSQL + Neo4j services running.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

CLEAN=""
THRESHOLD="0.85"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--clean)
      CLEAN="--clean"
      shift
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [-c|--clean] [--threshold NUM]"
      echo "  -c, --clean       Wipe all existing data before ingesting"
      echo "  --threshold NUM   Consolidation similarity threshold (default: 0.85)"
      exit 1
      ;;
  esac
done

python -m ingestion.ingest -d documents --chunk-size 1500 --chunk-overlap 250 $CLEAN
python -m ingestion.consolidate_graph --threshold "$THRESHOLD"