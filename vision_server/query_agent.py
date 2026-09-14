"""Natural language query agent — 32B model generates Cypher, self-corrects on failure.

Usage:
    python query_agent.py                        # interactive mode
    python query_agent.py "who was here at 8am?"  # single question
"""

import sys
import os
import json
from datetime import datetime
from ollama import Client
from neo4j_store import get_driver
from embeddings import get_embedding
from config import OLLAMA_REASONING_HOST, REASONING_MODEL

_client = Client(host=OLLAMA_REASONING_HOST)

GRAPH_SCHEMA = """
## Neo4j Graph Schema

### Nodes and Properties:
1. Camera {camera_id: STRING}
2. Observation {id: STRING, timestamp: DATETIME, trigger_type: STRING, yolo_class: STRING, yolo_confidence: FLOAT, brightness: FLOAT, motion_pct: FLOAT, image_path: STRING, processing_time_ms: INT, frame_num: INT}
3. Species {name: STRING}  -- values: "person", "heartbeat", "unknown", "wildlife_motion", etc.
4. TimeWindow {hour_key: STRING, date: STRING, hour: INT}  -- hour_key format: "YYYY-MM-DD_HH" e.g. "2026-09-13_11"
5. Description {text: STRING, embedding: LIST<FLOAT>}  -- VLM-generated scene description
6. Entity {name: STRING, type: STRING, attributes: LIST<STRING>}  -- type: "person", "animal", "vehicle", "object", "location"

### Relationships:
- (Camera)-[:CAPTURED]->(Observation)
- (Observation)-[:CLASSIFIED_AS]->(Species)
- (Observation)-[:OCCURRED_DURING]->(TimeWindow)
- (Observation)-[:DESCRIBED_BY]->(Description)
- (Observation)-[:CONTAINS]->(Entity)
- (Entity)-[:NEAR|INSIDE|WEARING|HOLDING|DOING|PART_OF]->(Entity)

### IMPORTANT Cypher Rules:
- ALWAYS use OPTIONAL MATCH for Description, Species, Entity joins (they may not exist for all observations)
- NEVER use pattern expressions in RETURN — use OPTIONAL MATCH before RETURN
- For time queries: MATCH on TimeWindow.hour_key or filter Observation.timestamp
- For text search: use toLower(d.text) CONTAINS toLower($keyword)
- For entity search: search e.name and e.attributes with toLower() CONTAINS
- ALWAYS RETURN: obs.timestamp, obs.image_path, description text, species name
- ALWAYS ORDER BY obs.timestamp DESC
- ALWAYS LIMIT to 10 unless asked otherwise
- For vector search: CALL db.index.vector.queryNodes('description_embedding', $limit, $embedding) YIELD node, score

### Example Queries:

-- All observations at hour 11 on Sept 13:
MATCH (obs:Observation)-[:OCCURRED_DURING]->(tw:TimeWindow {hour_key: "2026-09-13_11"})
OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
WITH obs, d, s, collect(e.name) AS entities
RETURN obs.timestamp AS timestamp, s.name AS species, d.text AS description, obs.image_path AS image_path, entities
ORDER BY obs.timestamp DESC LIMIT 10

-- Find person wearing yellow:
MATCH (obs:Observation)-[:DESCRIBED_BY]->(d:Description)
WHERE toLower(d.text) CONTAINS "yellow"
OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
WITH obs, d, s, collect(e.name) AS entities
RETURN obs.timestamp AS timestamp, s.name AS species, d.text AS description, obs.image_path AS image_path, entities
ORDER BY obs.timestamp DESC LIMIT 10

-- All people detected:
MATCH (obs:Observation)-[:CONTAINS]->(e:Entity {type: "person"})
OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
RETURN obs.timestamp AS timestamp, e.name AS entity, e.attributes AS attributes, d.text AS description, obs.image_path AS image_path
ORDER BY obs.timestamp DESC LIMIT 10

-- Summary stats:
MATCH (obs:Observation)
OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
RETURN s.name AS species, count(obs) AS count
ORDER BY count DESC

-- Recent activity:
MATCH (obs:Observation)
OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
RETURN obs.timestamp AS timestamp, s.name AS species, d.text AS description, obs.image_path AS image_path
ORDER BY obs.timestamp DESC LIMIT 10
"""


def generate_cypher(question, error_context=None):
    """Use 32B model to generate Cypher query. If error_context is provided, it's a retry."""
    today = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().hour

    messages = [
        {"role": "system", "content": f"""You are an expert Neo4j Cypher query generator for a wildlife/activity monitoring system.

Today is {today}, current hour is {current_hour}.

{GRAPH_SCHEMA}

RESPOND WITH ONLY A JSON OBJECT — no explanations, no markdown:
{{"cypher": "YOUR CYPHER QUERY", "params": {{}}, "needs_embedding": false}}

If the query needs semantic/vector search, set needs_embedding: true and include "search_text" in params:
{{"cypher": "CALL db.index.vector.queryNodes('description_embedding', $limit, $embedding) YIELD node, score MATCH (obs:Observation)-[:DESCRIBED_BY]->(node) OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species) RETURN obs.timestamp AS timestamp, s.name AS species, node.text AS description, obs.image_path AS image_path, score ORDER BY score DESC", "params": {{"limit": 5}}, "needs_embedding": true, "search_text": "the search phrase"}}"""},
        {"role": "user", "content": question},
    ]

    if error_context:
        messages.append({"role": "assistant", "content": error_context["previous_response"]})
        messages.append({"role": "user", "content": f"That query failed with error:\n{error_context['error']}\n\nFix the Cypher query. Return ONLY the corrected JSON object."})

    response = _client.chat(model=REASONING_MODEL, messages=messages)
    text = response["message"]["content"].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end]), text
    return None, text


def run_cypher(cypher, params=None):
    """Execute Cypher query."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        records = []
        for record in result:
            row = {}
            for key in record.keys():
                val = record[key]
                if hasattr(val, "isoformat"):
                    val = val.isoformat()
                row[key] = val
            records.append(row)
        return records


def explain_results(question, results, cypher):
    """Use 32B to explain results in natural language."""
    if not results:
        return "No matching observations found in the database."

    results_text = json.dumps(results[:10], indent=2, default=str)
    response = _client.chat(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": "You explain wildlife/activity monitoring data. Be concise and specific. Mention timestamps, descriptions, clothing, actions. Use markdown formatting. If image paths are available, mention them."},
            {"role": "user", "content": f"Question: {question}\n\nCypher: {cypher}\n\nResults ({len(results)} rows):\n{results_text}\n\nExplain these results clearly and concisely."},
        ],
    )
    return response["message"]["content"]


def ask(question, max_retries=2):
    """Full pipeline: generate Cypher -> execute -> self-correct if needed -> explain."""
    print(f"\n{'=' * 60}")
    print(f"  Q: {question}")
    print(f"{'=' * 60}")

    # Step 1: Generate Cypher
    print("\n  [1/3] Generating Cypher query...")
    parsed, raw_response = generate_cypher(question)

    if not parsed or "cypher" not in parsed:
        print(f"  ERROR: Could not generate query. Raw: {raw_response[:200]}")
        return None

    cypher = parsed["cypher"]
    params = parsed.get("params", {})

    if parsed.get("needs_embedding"):
        search_text = parsed.get("search_text", question)
        print(f"  Embedding: \"{search_text}\"")
        params["embedding"] = get_embedding(search_text)
        if "limit" not in params:
            params["limit"] = 5

    print(f"  Cypher: {cypher}")

    # Step 2: Execute with self-correction
    results = None
    for attempt in range(max_retries + 1):
        print(f"\n  [2/3] Running query (attempt {attempt + 1})...")
        try:
            results = run_cypher(cypher, params)
            print(f"  Results: {len(results)} rows")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"  Query failed: {error_msg}")

            if attempt < max_retries:
                print(f"  Self-correcting...")
                parsed, raw_response = generate_cypher(question, error_context={
                    "previous_response": json.dumps({"cypher": cypher, "params": {k: v for k, v in params.items() if k != "embedding"}}),
                    "error": error_msg,
                })
                if parsed and "cypher" in parsed:
                    cypher = parsed["cypher"]
                    params = parsed.get("params", {})
                    if parsed.get("needs_embedding"):
                        search_text = parsed.get("search_text", question)
                        params["embedding"] = get_embedding(search_text)
                        if "limit" not in params:
                            params["limit"] = 5
                    print(f"  New Cypher: {cypher}")
                else:
                    print(f"  Could not generate corrected query.")
                    break
            else:
                print(f"  Max retries reached.")

    if results is None:
        results = []

    # Show raw results
    if results:
        print("\n  --- Data ---")
        for i, row in enumerate(results[:5], 1):
            ts = row.get("timestamp", "?")
            desc = str(row.get("description", ""))[:80]
            species = row.get("species", "")
            print(f"  {i}. [{ts}] {species or ''} - {desc}...")
            if row.get("image_path"):
                print(f"     Image: {row['image_path']}")

    # Step 3: Explain
    print("\n  [3/3] Generating answer...")
    answer = explain_results(question, results, cypher)
    print(f"\n  {'-' * 56}")
    print(f"  ANSWER:\n  {answer}")
    print(f"  {'-' * 56}")

    return {"question": question, "cypher": cypher, "results": results, "answer": answer}


def interactive():
    """Interactive question loop."""
    print("=" * 60)
    print("  WILDLIFE MONITOR - Query Agent (32B)")
    print("  Ask questions about what the camera has seen.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        ask(question)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        interactive()
