from enum import Enum


class MetricEvent(Enum):
    REQUEST = "REQUEST"
    HIT = "HIT"
    MISS = "MISS"
    RESPONSE_REUSED = "RESPONSE_REUSED"
    RETRIEVAL_REUSED = "RETRIEVAL_REUSED"