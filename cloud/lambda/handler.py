"""Lambda function to process AntVision events from IoT Core."""

import json
import os
import boto3
from datetime import datetime

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "antvision-metrics")
S3_BUCKET = os.environ.get("S3_BUCKET", "antvision-data-dev")


def lambda_handler(event, context):
    table = dynamodb.Table(TABLE_NAME)
    event_type = event.get("event_type", "unknown")

    if event_type == "metrics_snapshot":
        item = {
            "experiment_id": event["experiment_id"],
            "timestamp": event["timestamp"],
            "device_id": event["device_id"],
            "event_type": event_type,
            "ant_count": event["metrics"]["ant_count"],
            "zone_occupancy": event["metrics"]["zone_occupancy"],
            "avg_speed": str(event["metrics"]["avg_speed"]),
        }
        table.put_item(Item=item)

    elif event_type in ("food_zone_enter", "food_zone_exit"):
        item = {
            "experiment_id": event["experiment_id"],
            "timestamp": event["timestamp"],
            "device_id": event["device_id"],
            "event_type": event_type,
            "ant_id": event["ant_id"],
            "x": event["x"],
            "y": event["y"],
            "zone": event.get("zone", "unknown"),
        }
        if "speed" in event:
            item["speed"] = str(event["speed"])
        table.put_item(Item=item)

    elif event_type == "experiment_end":
        key = f"{event['experiment_id']}/summary_{event['timestamp']}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(event, indent=2),
            ContentType="application/json",
        )

    return {"statusCode": 200, "event_type": event_type}
