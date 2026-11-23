"""
validate_compressed_body.py — compressed_body.py 가 압축한 요청을 일반 JSON 요청과 같은 payload 로 되돌리는지.

    python validation/validate_compressed_body.py      (backend-python 에서; fastapi · httpx 필요)

앱은 1MB 가 넘는 분석 요청을 Content-Type: application/vnd.skari.json+gzip, data = {"$columns": {...}}
로 보냅니다(statistica-frontend src/lib/analysis-engine/request-columns.ts). 여기서는
  1. 열 단위 자료 → 행 목록이 원래 행과 같은지(빈 칸 · 한글 열 이름 · 섞인 형식)
  2. 실제 FastAPI 라우트에서 압축 요청 · 일반 요청이 같은 payload 가 되는지
  3. 깨진 gzip 은 400, 풀어서 너무 큰 것은 413
을 봅니다. 실패하면 0 이 아닌 값으로 끝납니다.
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import compressed_body as cb  # noqa: E402

fails = 0


def ok(cond, msg):
    global fails
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        fails += 1


def columnar(rows):
    """앱의 toColumnar 와 같은 모양 — 열은 처음 나온 순서, 없는 칸은 null."""
    names = list(dict.fromkeys(k for r in rows for k in r))
    return {"$columns": {c: [r.get(c) for r in rows] for c in names}}


rows = [
    {"점수 (1~5)": 3, "grp": "A", "x": 1.5, "flag": True},
    {"점수 (1~5)": None, "grp": "다", "flag": False},
    {"점수 (1~5)": 5, "grp": 'O\'Neil "q"', "x": None, "extra": "only here"},
]
body = {"variables": ["점수 (1~5)", "x"], "params": {"group": "grp"}, "data": rows}
packed = {**body, "data": columnar(rows)}

# 1. 열 단위 → 행 목록
got = cb.expand_columns(json.loads(json.dumps(packed)))
same_cells = all(a.get(c) == b.get(c) for a, b in zip(rows, got["data"]) for c in set(a) | set(b))
ok(len(got["data"]) == 3 and same_cells, "columns expand to the same rows (missing cells read as None)")
ok(list(got["data"][0].keys()) == ["점수 (1~5)", "grp", "x", "flag", "extra"], "column order is kept")
ok({k: v for k, v in got.items() if k != "data"} == {k: v for k, v in body.items() if k != "data"}, "other fields are untouched")
ok(cb.expand_columns(body) is body, "a normal row-list body is returned as is")
for bad, why in (({"data": {"$columns": {"a": [1, 2], "b": [1]}}}, "uneven columns"), ({"data": {"$columns": [1, 2]}}, "not a dict")):
    try:
        cb.expand_columns(bad)
        ok(False, f"rejects {why}")
    except ValueError:
        ok(True, f"rejects {why}")

# 2. 실제 라우트에서 — main.py 의 _route 와 같은 읽기
app = FastAPI()


@app.post("/echo")
async def echo(request: Request):
    return await cb.read_payload(request)


client = TestClient(app)
plain = client.post("/echo", content=json.dumps(body), headers={"Content-Type": "application/json"})
zipped = client.post("/echo", content=gzip.compress(json.dumps(packed).encode()),
                     headers={"Content-Type": "application/vnd.skari.json+gzip"})
ok(plain.status_code == 200 and zipped.status_code == 200, "both requests are accepted")
norm = lambda d: {**d, "data": [{c: r.get(c) for c in ["점수 (1~5)", "grp", "x", "flag", "extra"]} for r in d["data"]]}
ok(norm(plain.json()) == norm(zipped.json()), "compressed and plain requests give the same payload")
ok(client.post("/echo", content=b"\x1f\x8bnot really gzip", headers={"Content-Type": "application/vnd.skari.json+gzip"}).status_code == 400,
   "a broken gzip body is 400")
truncated = gzip.compress(json.dumps(packed).encode())[:-12]
ok(client.post("/echo", content=truncated, headers={"Content-Type": "application/vnd.skari.json+gzip"}).status_code == 400,
   "a truncated gzip body is 400")

# 3. 압축 폭탄 — 작은 gzip 이 아주 크게 풀리는 것
bomb =gzip.compress(b'{"data": "' + b"a" * (2 * 1024 * 1024) + b'"}')
try:
    cb.gunzip_limited(bomb, max_bytes=1024 * 1024)
    ok(False, "stops past the unpacked limit")
except ValueError as e:
    ok(str(e) == "too large", "stops past the unpacked limit")
ok(len(bomb) < 20 * 1024, f"(the test bomb is {len(bomb)} bytes compressed)")

print(f"\n{fails} FAILED" if fails else "\nALL OK")
sys.exit(1 if fails else 0)
