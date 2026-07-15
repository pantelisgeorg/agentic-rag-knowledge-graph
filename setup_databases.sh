#!/usr/bin/env bash
# Local database setup for Agentic RAG with Knowledge Graph (NO Docker).
# Run from the project root with:  sudo bash setup_databases.sh
#
# Installs: PostgreSQL 16 + pgvector, Neo4j 5.x
# Creates:  PostgreSQL DB 'agentic_rag_db' (user raguser), applies sql/schema.sql
# Sets:     Neo4j initial password
set -euo pipefail

cd "$(dirname "$0")"

DB_USER=raguser
DB_PASS=ragpass123
DB_NAME=agentic_rag_db
NEO4J_PASS=neo4jragpass

echo "==> [1/7] Installing PostgreSQL 16 + pgvector + client"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    postgresql postgresql-contrib postgresql-16-pgvector postgresql-client-16

echo "==> [2/7] Starting PostgreSQL"
systemctl enable --now postgresql

echo "==> [3/7] Creating PostgreSQL role + database"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$do\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}' SUPERUSER;
  ELSE
    ALTER ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}' SUPERUSER;
  END IF;
END
\$do\$;
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec
SQL

echo "==> [4/7] Applying sql/schema.sql to ${DB_NAME}"
PGPASSWORD=${DB_PASS} psql -h localhost -U ${DB_USER} -d ${DB_NAME} \
    -v ON_ERROR_STOP=1 -f sql/schema.sql
echo "    Schema applied (vector dim = 768 for nomic-embed-text-v2-moe)."

echo "==> [5/7] Installing OpenJDK 21 (Neo4j runtime dependency)"
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-21-jre-headless

echo "==> [6/7] Adding Neo4j apt repository + installing Neo4j"
wget -qO - https://debian.neo4j.com/neotechnology.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
echo 'deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest' \
    > /etc/apt/sources.list.d/neo4j.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y neo4j

echo "==> [7/7] Configuring Neo4j password + starting service"
systemctl stop neo4j 2>/dev/null || true
if neo4j-admin dbms set-initial-password "${NEO4J_PASS}" 2>/dev/null; then
    echo "    Initial Neo4j password set."
else
    echo "    set-initial-password unavailable (DB already initialized). Trying cypher-shell change..."
    systemctl start neo4j
    sleep 8
    cypher-shell -u neo4j -p neo4j \
        "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO '${NEO4J_PASS}';" 2>/dev/null \
        || (cypher-shell -u neo4j -p "${NEO4J_PASS}" "RETURN 1;" >/dev/null 2>&1 \
            && echo "    Neo4j password already correct.")
fi
systemctl enable --now neo4j

echo ""
echo "==> Verification"
sleep 5
echo -n "PostgreSQL service: "; systemctl is-active postgresql
echo -n "Neo4j service:      "; systemctl is-active neo4j
echo -n "DB connection:      "
PGPASSWORD=${DB_PASS} psql -h localhost -U ${DB_USER} -d ${DB_NAME} -tAc "SELECT 'ok';"
echo ""
echo "Done."
echo "  PostgreSQL: postgresql://${DB_USER}:****@localhost:5432/${DB_NAME}"
echo "  Neo4j:      bolt://localhost:7687  (user neo4j / password ${NEO4J_PASS})"
