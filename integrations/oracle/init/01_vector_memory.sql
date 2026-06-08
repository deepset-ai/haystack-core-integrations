-- Enable vector memory pool required for HNSW in-memory vector indexes.
-- This script runs at container startup via /container-entrypoint-initdb.d.
-- The primary mechanism is ORACLE_INIT_PARAMS=vector_memory_size=512M in
-- docker-compose.yml (writes to SPFILE at DB creation time).  This ALTER
-- SYSTEM acts as a belt-and-suspenders dynamic setter for pre-existing
-- database volumes where the SPFILE value may not have been written yet.
ALTER SYSTEM SET vector_memory_size = 512M SCOPE=BOTH;
