"""Neo4j graph database storage for wildlife vision events.

Stores observations as a graph:
  (Camera)-[:CAPTURED]->(Observation)-[:OBSERVED_AT]->(Location)
  (Observation)-[:CLASSIFIED_AS]->(Species)
  (Observation)-[:DESCRIBED_BY]->(Description)  [with vector embedding]
  (Observation)-[:OCCURRED_DURING]->(TimeWindow)

Vector index on Description nodes enables semantic search like:
  "what happened near the door around 8am?"
"""

import os
from datetime import datetime, timezone

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "wildlife2026")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        _driver.verify_connectivity()
        print(f"[neo4j] Connected to {NEO4J_URI}")
    return _driver


def init_schema():
    """Create constraints, indexes, and vector index on first run."""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT obs_id IF NOT EXISTS "
            "FOR (o:Observation) REQUIRE o.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT cam_id IF NOT EXISTS "
            "FOR (c:Camera) REQUIRE c.camera_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT species_name IF NOT EXISTS "
            "FOR (s:Species) REQUIRE s.name IS UNIQUE"
        )
        session.run(
            "CREATE INDEX obs_timestamp IF NOT EXISTS "
            "FOR (o:Observation) ON (o.timestamp)"
        )
        session.run(
            "CREATE VECTOR INDEX description_embedding IF NOT EXISTS "
            "FOR (d:Description) ON (d.embedding) "
            "OPTIONS {indexConfig: {"
            " `vector.dimensions`: 384,"
            " `vector.similarity_function`: 'cosine'"
            "}}"
        )
    print("[neo4j] Schema initialized (constraints + vector index)")


def store_event(event):
    """Store a vision event in the graph.

    event dict should have: timestamp, camera_id, trigger_type,
    yolo_class, yolo_confidence, brightness, motion_pct, description,
    embedding (list of 384 floats), image_path, processing_time_ms.
    """
    driver = get_driver()

    ts = event.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        dt = datetime.now(timezone.utc)

    hour_key = dt.strftime("%Y-%m-%d_%H")
    obs_id = f"{event.get('camera_id', 'unknown')}_{dt.strftime('%Y%m%dT%H%M%S')}_{event.get('frame_num', 0)}"

    with driver.session() as session:
        session.run("""
            MERGE (cam:Camera {camera_id: $camera_id})

            MERGE (species:Species {name: $species_name})

            MERGE (tw:TimeWindow {hour_key: $hour_key})
            ON CREATE SET tw.date = $date, tw.hour = $hour

            CREATE (obs:Observation {
                id: $obs_id,
                timestamp: datetime($timestamp),
                trigger_type: $trigger_type,
                yolo_class: $yolo_class,
                yolo_confidence: $yolo_confidence,
                brightness: $brightness,
                motion_pct: $motion_pct,
                image_path: $image_path,
                processing_time_ms: $processing_time_ms,
                frame_num: $frame_num
            })

            CREATE (desc:Description {
                text: $description,
                embedding: $embedding
            })

            CREATE (cam)-[:CAPTURED]->(obs)
            CREATE (obs)-[:CLASSIFIED_AS]->(species)
            CREATE (obs)-[:OCCURRED_DURING]->(tw)
            CREATE (obs)-[:DESCRIBED_BY]->(desc)
        """, {
            "camera_id": event.get("camera_id", "pi-default"),
            "species_name": event.get("yolo_class", "unknown") or "unknown",
            "hour_key": hour_key,
            "date": dt.strftime("%Y-%m-%d"),
            "hour": dt.hour,
            "obs_id": obs_id,
            "timestamp": dt.isoformat(),
            "trigger_type": event.get("trigger_type", ""),
            "yolo_class": event.get("yolo_class", ""),
            "yolo_confidence": float(event.get("yolo_confidence", 0)),
            "brightness": float(event.get("brightness", 0)),
            "motion_pct": float(event.get("motion_pct", 0)),
            "image_path": event.get("image_path", ""),
            "processing_time_ms": int(event.get("processing_time_ms", 0)),
            "frame_num": int(event.get("frame_num", 0)),
            "description": event.get("description", ""),
            "embedding": event.get("embedding", []),
        })

    return obs_id


def search_similar(query_embedding, limit=5):
    """Find observations with descriptions semantically similar to query."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('description_embedding', $limit, $embedding)
            YIELD node, score
            MATCH (obs:Observation)-[:DESCRIBED_BY]->(node)
            OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
            RETURN obs.id AS id,
                   obs.timestamp AS timestamp,
                   obs.trigger_type AS trigger,
                   s.name AS species,
                   node.text AS description,
                   obs.image_path AS image_path,
                   score
            ORDER BY score DESC
        """, {"embedding": query_embedding, "limit": limit})
        return [dict(record) for record in result]


def search_by_time(start_hour, end_hour=None):
    """Find observations in a time range (hour_key format: YYYY-MM-DD_HH)."""
    driver = get_driver()
    with driver.session() as session:
        if end_hour:
            result = session.run("""
                MATCH (tw:TimeWindow)
                WHERE tw.hour_key >= $start AND tw.hour_key <= $end
                MATCH (obs:Observation)-[:OCCURRED_DURING]->(tw)
                OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
                OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
                RETURN obs.id AS id,
                       obs.timestamp AS timestamp,
                       obs.trigger_type AS trigger,
                       s.name AS species,
                       d.text AS description,
                       obs.image_path AS image_path
                ORDER BY obs.timestamp
            """, {"start": start_hour, "end": end_hour})
        else:
            result = session.run("""
                MATCH (obs:Observation)-[:OCCURRED_DURING]->(tw:TimeWindow {hour_key: $hour_key})
                OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
                OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
                RETURN obs.id AS id,
                       obs.timestamp AS timestamp,
                       obs.trigger_type AS trigger,
                       s.name AS species,
                       d.text AS description,
                       obs.image_path AS image_path
                ORDER BY obs.timestamp
            """, {"hour_key": start_hour})
        return [dict(record) for record in result]


def get_stats():
    """Get summary stats from the graph."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (o:Observation)
            WITH count(o) AS total_obs
            OPTIONAL MATCH (s:Species)
            WITH total_obs, collect(s.name) AS species
            OPTIONAL MATCH (c:Camera)
            RETURN total_obs,
                   species,
                   collect(c.camera_id) AS cameras
        """)
        record = result.single()
        if record:
            return dict(record)
        return {"total_obs": 0, "species": [], "cameras": []}


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
