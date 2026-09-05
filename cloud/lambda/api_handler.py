"""API Lambda for the dashboard — serves metrics and image URLs from AWS."""

import json
import os
import boto3
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "antvision-metrics")
S3_BUCKET = os.environ.get("S3_BUCKET", "antvision-data-dev")


def lambda_handler(event, context):
    path = event.get("rawPath", event.get("path", "/"))
    params = event.get("queryStringParameters") or {}
    experiment_id = params.get("experiment", "exp001")

    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
    }

    if path == "/api/metrics":
        return _get_metrics(experiment_id, headers)
    elif path == "/api/events":
        return _get_events(experiment_id, headers)
    elif path == "/api/images":
        return _get_images(experiment_id, headers)
    elif path == "/api/latest":
        return _get_latest(experiment_id, headers)
    else:
        return {"statusCode": 404, "headers": headers, "body": json.dumps({"error": "not found"})}


def _get_metrics(experiment_id, headers):
    table = dynamodb.Table(TABLE_NAME)
    response = table.query(
        KeyConditionExpression="experiment_id = :eid AND begins_with(#ts, :prefix)",
        ExpressionAttributeNames={"#ts": "timestamp"},
        ExpressionAttributeValues={
            ":eid": experiment_id,
            ":prefix": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        ScanIndexForward=False,
        Limit=50,
    )
    items = response.get("Items", [])

    snapshots = [i for i in items if i.get("event_type") == "metrics_snapshot"]
    zone_entries = [i for i in items if i.get("event_type") == "food_zone_enter"]
    zone_exits = [i for i in items if i.get("event_type") == "food_zone_exit"]

    latest = snapshots[0] if snapshots else {}
    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "ant_count": int(latest.get("ant_count", 0)),
            "zone_occupancy": int(latest.get("zone_occupancy", 0)),
            "avg_speed": float(latest.get("avg_speed", 0)),
            "total_zone_entries": len(zone_entries),
            "total_zone_exits": len(zone_exits),
            "last_updated": latest.get("timestamp", ""),
        }),
    }


def _get_events(experiment_id, headers):
    table = dynamodb.Table(TABLE_NAME)
    response = table.query(
        KeyConditionExpression="experiment_id = :eid",
        ExpressionAttributeValues={":eid": experiment_id},
        ScanIndexForward=False,
        Limit=100,
    )
    items = response.get("Items", [])
    for item in items:
        for key, val in item.items():
            if hasattr(val, 'as_integer_ratio'):
                item[key] = float(val)

    return {"statusCode": 200, "headers": headers, "body": json.dumps(items)}


def _get_images(experiment_id, headers):
    folders = ["heartbeat", "motion", "ant_detected", "ant_arrival", "zone_event",
                "wildlife_motion"]
    images = []

    for folder in folders:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{experiment_id}/{folder}/",
            MaxKeys=20,
        )
        for obj in response.get("Contents", []):
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": obj["Key"]},
                ExpiresIn=3600,
            )
            filename = obj["Key"].split("/")[-1]
            images.append({
                "key": obj["Key"],
                "url": url,
                "filename": filename,
                "trigger": folder,
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })

    images.sort(key=lambda x: x["last_modified"], reverse=True)
    return {"statusCode": 200, "headers": headers, "body": json.dumps(images[:50])}


def _get_latest(experiment_id, headers):
    all_objects = []
    for folder in ["heartbeat", "motion", "ant_detected", "ant_arrival", "zone_event",
                    "wildlife_motion"]:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{experiment_id}/{folder}/",
            MaxKeys=10,
        )
        all_objects.extend(response.get("Contents", []))

    if not all_objects:
        return {"statusCode": 200, "headers": headers, "body": json.dumps({"image": None})}

    latest = max(all_objects, key=lambda x: x["LastModified"])
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": latest["Key"]},
        ExpiresIn=3600,
    )
    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "image": url,
            "key": latest["Key"],
            "timestamp": latest["LastModified"].isoformat(),
        }),
    }
