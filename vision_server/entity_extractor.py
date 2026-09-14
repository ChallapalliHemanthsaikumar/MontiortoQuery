"""Extract structured entities and relationships from VLM descriptions.

Takes the raw text description from Qwen VLM and extracts:
- Entities: person, animal, vehicle, object, location
- Attributes: color, clothing, size, state
- Relationships: near, inside, wearing, holding, doing
"""

import json
from ollama import Client
from config import OLLAMA_REASONING_HOST, REASONING_MODEL

_client = Client(host=OLLAMA_REASONING_HOST)

EXTRACTION_PROMPT = """You are an entity extraction system. Given a scene description from a wildlife/activity camera, extract structured entities and relationships.

Return ONLY valid JSON with this exact format:
{
  "entities": [
    {"type": "person|animal|vehicle|object|location", "name": "short label", "attributes": ["attr1", "attr2"]}
  ],
  "relationships": [
    {"from": "entity name", "relation": "near|inside|wearing|holding|doing|part_of", "to": "entity name or action"}
  ]
}

Rules:
- Keep entity names short (2-3 words max)
- Attributes include: colors, clothing, size, material, state
- Extract spatial relationships (near, inside, on)
- Extract action relationships (doing, holding, wearing)
- If no people/animals, still extract objects and locations
- Return empty arrays if nothing meaningful to extract"""


def extract_entities(description, yolo_class=""):
    """Extract entities and relationships from a VLM description."""
    hint = f" YOLO detected: {yolo_class}." if yolo_class else ""

    try:
        response = _client.chat(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract entities from this scene description:{hint}\n\n\"{description}\""},
            ],
        )

        text = response["message"]["content"].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            for e in entities:
                if "type" not in e:
                    e["type"] = "object"
                if "name" not in e:
                    e["name"] = "unknown"
                if "attributes" not in e:
                    e["attributes"] = []
            return {"entities": entities, "relationships": relationships}
    except Exception as e:
        print(f"  [entity-extractor] Failed: {e}")

    return {"entities": [], "relationships": []}
