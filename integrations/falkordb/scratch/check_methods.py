import falkordb
import os

host = os.environ.get("FALKORDB_HOST", "localhost")
port = int(os.environ.get("FALKORDB_PORT", "6379"))

try:
    client = falkordb.FalkorDB(host=host, port=port)
    print("Client methods:", [m for m in dir(client) if not m.startswith("_")])
    
    graph = client.select_graph("dummy_test_graph")
    print("Graph methods:", [m for m in dir(graph) if not m.startswith("_")])
except Exception as e:
    print(f"Error connecting: {e}")
