# feedback.py — one JSON file per feedback click, in S3 (AWS) or ./feedback (local).

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3


def _feedback_location():
    # Feedback lives next to the documents: s3://<bucket>/documents/ -> s3://<bucket>/feedback/
    s3_uri = os.getenv("FEEDBACK_S3_URI") or os.getenv("KNOWLEDGE_BASE_S3_URI")
    if not s3_uri:
        return None, Path("feedback")
    bucket = s3_uri.removeprefix("s3://").split("/", 1)[0]
    return bucket, "feedback"


class FeedbackStore:
    def __init__(self):
        self.bucket, self.prefix = _feedback_location()
        self.s3 = boto3.client("s3") if self.bucket else None

    def save(self, event):
        now = datetime.now(timezone.utc)
        event = {"timestamp": now.isoformat(timespec="seconds"), **event}
        name = f"{now:%Y/%m/%d}/{uuid.uuid4()}.json"
        body = json.dumps(event, ensure_ascii=False, indent=2)
        if self.s3:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/{name}",
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            target = self.prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
