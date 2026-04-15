-- Enable vector memory pool required for HNSW in-memory vector indexes.
-- Runs as SYS against the CDB on first database startup.
ALTER SYSTEM SET vector_memory_size = 512M;
