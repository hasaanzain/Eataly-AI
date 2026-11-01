# events.py — query and feedback records, one JSON file each, in S3 (AWS) or ./logs (local).
#
#   queries/YYYY/MM/DD/<message_id>.json   every question: rewrite, route, retrieved sources, answer, latency
#   feedback/YYYY/MM/DD/<message_id>.json  the rating for that answer (a later click on the same day replaces it)
#
# Each file is a single line of JSON so the Athena tables in infra/aws/foundation.yaml can read it.

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3


KINDS = ("queries", "feedback")


def _bucket():
    # Records live next to the documents: s3://<bucket>/documents/ -> s3://<bucket>/queries/ and /feedback/
    s3_uri = os.getenv("EVENTS_S3_URI") or os.getenv("KNOWLEDGE_BASE_S3_URI")
    return s3_uri.removeprefix("s3://").split("/", 1)[0] if s3_uri else None


class EventStore:
    def __init__(self, bucket=None, local_dir=Path("logs")):
        self.bucket = bucket if bucket is not None else _bucket()
        self.s3 = boto3.client("s3") if self.bucket else None
        self.local_dir = Path(local_dir)

    def save(self, kind, message_id, event):
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        now = datetime.now(timezone.utc)
        event = {"created_at": now.isoformat(timespec="seconds"), "message_id": message_id, **event}
        key = f"{kind}/{now:%Y/%m/%d}/{message_id}.json"
        body = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        if self.s3:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=body.encode("utf-8"),
                               ContentType="application/json")
        else:
            target = self.local_dir / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return key
