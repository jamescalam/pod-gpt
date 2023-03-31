from pydantic import BaseModel
from typing import Optional


class Metadata(BaseModel):
    title: str
    channel_id: str
    published: str
    source: str
    chunk: int

class Record(BaseModel):
    id: str
    text: str
    metadata: Metadata

class VideoRecord(BaseModel):
    video_id: str
    channel_id: str
    title: str
    published: str
    source: str
    transcript: Optional[str] = None