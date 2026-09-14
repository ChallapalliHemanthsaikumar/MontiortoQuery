"""Wildlife Monitor Dashboard — Streamlit chat UI with image display.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import os
import sys
import json
from datetime import datetime
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vision_server"))

from neo4j_store import get_driver, search_similar
from embeddings import get_embedding
from ollama import Client
from config import OLLAMA_REASONING_HOST, REASONING_MODEL

_client = Client(host=OLLAMA_REASONING_HOST)

st.set_page_config(page_title="Wildlife Monitor", page_icon="📷", layout="wide")

st.title("Wildlife Activity Monitor")
st.caption("Ask questions about what the camera has seen")


@st.cache_resource
def get_neo4j():
    return get_driver()


def query_neo4j(cypher, params=None):
    driver = get_neo4j()
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


def get_all_observations(limit=50):
    return query_neo4j("""
        MATCH (obs:Observation)
        OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
        OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
        OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
        WITH obs, d, s, collect({name: e.name, type: e.type, attributes: e.attributes}) AS entities
        RETURN obs.id AS id, obs.timestamp AS timestamp, obs.trigger_type AS trigger,
               obs.brightness AS brightness, obs.image_path AS image_path,
               s.name AS species, d.text AS description, entities
        ORDER BY obs.timestamp DESC LIMIT $limit
    """, {"limit": limit})


def search_by_text(keyword, limit=20):
    return query_neo4j("""
        MATCH (obs:Observation)-[:DESCRIBED_BY]->(d:Description)
        WHERE toLower(d.text) CONTAINS toLower($keyword)
        OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
        OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
        WITH obs, d, s, collect({name: e.name, type: e.type, attributes: e.attributes}) AS entities
        RETURN obs.id AS id, obs.timestamp AS timestamp, obs.trigger_type AS trigger,
               obs.image_path AS image_path, s.name AS species, d.text AS description, entities
        ORDER BY obs.timestamp DESC LIMIT $limit
    """, {"keyword": keyword, "limit": limit})


def search_by_entity(entity_type=None, keyword=None, limit=20):
    if entity_type and keyword:
        return query_neo4j("""
            MATCH (obs:Observation)-[:CONTAINS]->(e:Entity)
            WHERE e.type = $type AND (toLower(e.name) CONTAINS toLower($keyword)
                  OR ANY(attr IN e.attributes WHERE toLower(attr) CONTAINS toLower($keyword)))
            OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
            RETURN obs.id AS id, obs.timestamp AS timestamp, obs.image_path AS image_path,
                   e.name AS entity, e.type AS entity_type, e.attributes AS attributes,
                   d.text AS description
            ORDER BY obs.timestamp DESC LIMIT $limit
        """, {"type": entity_type, "keyword": keyword, "limit": limit})
    elif entity_type:
        return query_neo4j("""
            MATCH (obs:Observation)-[:CONTAINS]->(e:Entity)
            WHERE e.type = $type
            OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
            RETURN obs.id AS id, obs.timestamp AS timestamp, obs.image_path AS image_path,
                   e.name AS entity, e.type AS entity_type, e.attributes AS attributes,
                   d.text AS description
            ORDER BY obs.timestamp DESC LIMIT $limit
        """, {"type": entity_type, "limit": limit})
    else:
        return query_neo4j("""
            MATCH (obs:Observation)-[:CONTAINS]->(e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($keyword)
                  OR ANY(attr IN e.attributes WHERE toLower(attr) CONTAINS toLower($keyword))
            OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
            RETURN obs.id AS id, obs.timestamp AS timestamp, obs.image_path AS image_path,
                   e.name AS entity, e.type AS entity_type, e.attributes AS attributes,
                   d.text AS description
            ORDER BY obs.timestamp DESC LIMIT $limit
        """, {"keyword": keyword, "limit": limit})


def search_by_time(hour, date=None, limit=20):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    hour_key = f"{date}_{hour:02d}"
    return query_neo4j("""
        MATCH (obs:Observation)-[:OCCURRED_DURING]->(tw:TimeWindow {hour_key: $hour_key})
        OPTIONAL MATCH (obs)-[:DESCRIBED_BY]->(d:Description)
        OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
        OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
        WITH obs, d, s, collect({name: e.name, type: e.type, attributes: e.attributes}) AS entities
        RETURN obs.id AS id, obs.timestamp AS timestamp, obs.trigger_type AS trigger,
               obs.image_path AS image_path, s.name AS species, d.text AS description, entities
        ORDER BY obs.timestamp DESC LIMIT $limit
    """, {"hour_key": hour_key, "limit": limit})


def search_semantic(query_text, limit=5):
    embedding = get_embedding(query_text)
    return query_neo4j("""
        CALL db.index.vector.queryNodes('description_embedding', $limit, $embedding)
        YIELD node, score
        MATCH (obs:Observation)-[:DESCRIBED_BY]->(node)
        OPTIONAL MATCH (obs)-[:CLASSIFIED_AS]->(s:Species)
        OPTIONAL MATCH (obs)-[:CONTAINS]->(e:Entity)
        WITH obs, node, score, s, collect({name: e.name, type: e.type}) AS entities
        RETURN obs.id AS id, obs.timestamp AS timestamp, obs.trigger_type AS trigger,
               obs.image_path AS image_path, s.name AS species,
               node.text AS description, entities, score
        ORDER BY score DESC
    """, {"embedding": embedding, "limit": limit})


def generate_and_run(question, max_retries=2):
    """Generate Cypher with 32B model, execute, self-correct on failure."""
    from query_agent import GRAPH_SCHEMA

    today = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().hour

    messages = [
        {"role": "system", "content": f"""You are an expert Neo4j Cypher query generator for a wildlife/activity monitoring system.

Today is {today}, current hour is {current_hour}.

{GRAPH_SCHEMA}

RESPOND WITH ONLY A JSON OBJECT:
{{"cypher": "YOUR CYPHER QUERY", "params": {{}}, "needs_embedding": false}}

For vector search, set needs_embedding: true and include search_text."""},
        {"role": "user", "content": question},
    ]

    for attempt in range(max_retries + 1):
        response = _client.chat(model=REASONING_MODEL, messages=messages)
        text = response["message"]["content"].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return [], "Could not generate query"

        parsed = json.loads(text[start:end])
        cypher = parsed.get("cypher", "")
        params = parsed.get("params", {})

        if parsed.get("needs_embedding"):
            search_text = parsed.get("search_text", question)
            params["embedding"] = get_embedding(search_text)
            if "limit" not in params:
                params["limit"] = 5

        try:
            results = query_neo4j(cypher, params)
            return results, cypher
        except Exception as e:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Query failed: {e}\nFix the Cypher. Return ONLY corrected JSON."})
            else:
                return [], f"Query failed after {max_retries + 1} attempts: {e}"

    return [], "Max retries exceeded"


def explain_results(question, results):
    if not results:
        return "No matching observations found."

    results_text = json.dumps(results[:10], indent=2, default=str)
    response = _client.chat(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": "You explain wildlife/activity monitoring results. Be concise. Mention times, what was seen. Use markdown for formatting."},
            {"role": "user", "content": f"Question: {question}\n\nData ({len(results)} results):\n{results_text}\n\nAnswer concisely."},
        ],
    )
    return response["message"]["content"]


def display_observation(obs):
    """Display a single observation with image."""
    col1, col2 = st.columns([1, 2])

    with col1:
        image_path = obs.get("image_path", "")
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, use_container_width=True)
        else:
            st.info("Image not available")

    with col2:
        ts = obs.get("timestamp", "Unknown time")
        trigger = obs.get("trigger", obs.get("trigger_type", ""))
        species = obs.get("species", "")

        badge = "🟢" if trigger == "heartbeat" else "🔴" if "person" in str(trigger) else "🟡"
        st.markdown(f"**{badge} {ts}** | `{trigger}` | {species}")

        desc = obs.get("description", "No description")
        st.markdown(f"_{desc}_")

        entities = obs.get("entities", [])
        if entities:
            ent_tags = [f"`{e['name']}` ({e.get('type', '')})" for e in entities if e.get("name")]
            if ent_tags:
                st.markdown("**Entities:** " + " | ".join(ent_tags))

    st.divider()


# Sidebar stats
with st.sidebar:
    st.header("System Status")
    try:
        stats = query_neo4j("MATCH (o:Observation) RETURN count(o) AS total")
        total = stats[0]["total"] if stats else 0
        st.metric("Total Observations", total)

        species = query_neo4j("MATCH (s:Species)<-[:CLASSIFIED_AS]-(o) RETURN s.name AS name, count(o) AS cnt ORDER BY cnt DESC")
        if species:
            st.subheader("Species")
            for s in species:
                st.write(f"**{s['name']}**: {s['cnt']}")

        entities = query_neo4j("MATCH (e:Entity) RETURN e.type AS type, count(e) AS cnt ORDER BY cnt DESC LIMIT 10")
        if entities:
            st.subheader("Entity Types")
            for e in entities:
                st.write(f"**{e['type']}**: {e['cnt']}")

    except Exception as e:
        st.error(f"Neo4j: {e}")

    st.divider()
    st.subheader("Quick Filters")
    if st.button("All Observations"):
        st.session_state.quick_query = "show me everything"
    if st.button("All People"):
        st.session_state.quick_query = "show me all people"
    if st.button("Recent Activity"):
        st.session_state.quick_query = "what happened in the last hour"
    if st.button("Animals"):
        st.session_state.quick_query = "any animals detected"

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "observations" in msg:
            for obs in msg["observations"]:
                display_observation(obs)

# Handle quick query from sidebar
if "quick_query" in st.session_state:
    prompt = st.session_state.pop("quick_query")
else:
    prompt = st.chat_input("Ask about what the camera has seen...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check if it's a casual message (not a data query)
        casual_words = ["hi", "hello", "hey", "thanks", "thank you", "bye", "help", "what can you do", "who are you"]
        is_casual = prompt.strip().lower().rstrip("!?.") in casual_words

        if is_casual:
            casual_response = _client.chat(
                model=REASONING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a friendly wildlife activity monitor assistant. You help users query camera observations stored in a graph database. You can answer questions about what the camera has seen — people, animals, time-based activity, descriptions, etc. Keep it short and friendly."},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = casual_response["message"]["content"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            results = None
        else:
            with st.spinner("Generating query and searching..."):
                results, cypher = generate_and_run(prompt)

        if results:
            with st.spinner("Generating answer..."):
                answer = explain_results(prompt, results)

            st.markdown(answer)
            st.divider()
            st.subheader(f"Observations ({len(results)})")
            for obs in results[:10]:
                display_observation(obs)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "observations": results[:10],
            })
        else:
            st.warning("No matching observations found.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "No matching observations found.",
            })
