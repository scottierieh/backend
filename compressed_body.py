"""
압축한 분석 요청 — 앱이 큰 요청(1MB 넘음)을 gzip 으로 보낼 때.

앱(statistica-frontend src/lib/analysis-engine/request-columns.ts)은 큰 요청을

    Content-Type: application/vnd.skari.json+gzip
    본문: gzip( {"...": ..., "data": {"$columns": {"열": [값, ...], ...}}} )

으로 보냅니다. 열 이름을 행마다 되풀이하지 않고 압축해서, 같은 자료가 보통 5~10배 작아집니다.
read_payload() 가 압축을 풀고 `data` 를 원래의 행 목록([{열: 값}, ...])으로 되돌려, 분석 스크립트는
바뀐 것을 모릅니다. 압축하지 않은 요청은 지금과 똑같이 request.json() 그대로입니다.

받는 크기는 지금처럼 BodySizeLimitMiddleware(SKARI_MAX_BODY_MB, 25MB)가 압축된 크기에 겁니다.
압축을 푼 크기는 SKARI_MAX_UNPACKED_MB(기본 160MB)까지 — 압축 폭탄 방지.
R 서버의 같은 처리: statistica-frontend r-backend/handlers/_compressed_body.R
"""
import json
import os
import zlib

from fastapi import HTTPException, Request

COMPRESSED_TYPE = "application/vnd.skari.json+gzip"
COLUMNS_KEY = "$columns"
MAX_UNPACKED_BYTES = int(float(os.environ.get("SKARI_MAX_UNPACKED_MB", "160")) * 1024 * 1024)


class _TooLarge(ValueError):
    pass


def is_compressed(content_type: str | None) -> bool:
    return (content_type or "").split(";")[0].strip().lower() == COMPRESSED_TYPE


def gunzip_limited(raw: bytes, max_bytes: int = MAX_UNPACKED_BYTES) -> bytes:
    """gzip 바이트를 풀되 max_bytes 를 넘으면 그 자리에서 멈춥니다. 잘린 · 깨진 gzip 은 ValueError."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = d.decompress(raw, max_bytes + 1)
    if len(out) > max_bytes or d.unconsumed_tail:
        raise _TooLarge("too large")
    if not d.eof:
        raise ValueError("truncated gzip body")
    return out


def expand_columns(payload):
    """payload["data"] 가 {"$columns": {열: [값...]}} 이면 행 목록으로. 아니면 그대로."""
    if not isinstance(payload, dict):
        return payload
    data = payload.get("data")
    if not (isinstance(data, dict) and list(data.keys()) == [COLUMNS_KEY]):
        return payload
    cols = data[COLUMNS_KEY]
    if not isinstance(cols, dict) or not all(isinstance(v, list) for v in cols.values()):
        raise ValueError("invalid column data")
    if len({len(v) for v in cols.values()}) > 1:
        raise ValueError("columns have different lengths")
    names = list(cols.keys())
    rows = [dict(zip(names, values)) for values in zip(*cols.values())]
    return {**payload, "data": rows}


async def read_payload(request: Request):
    """분석 요청 본문 → dict. 압축한 요청이면 풀고 열 단위 자료를 행 목록으로."""
    if not is_compressed(request.headers.get("content-type")):
        return await request.json()
    raw = await request.body()
    try:
        payload = json.loads(gunzip_limited(raw))
        return expand_columns(payload)
    except _TooLarge:
        raise HTTPException(status_code=413, detail="The request is too large after decompression. Use fewer variables or rows.")
    except (ValueError, zlib.error, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="The compressed request body could not be read.")
