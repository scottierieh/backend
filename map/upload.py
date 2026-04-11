"""
routers/upload.py
파일 업로드 & 파싱 엔드포인트

POST /api/upload   — CSV / TSV / Excel → rows + header 정보 반환
"""

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from map.utils import to_native

router = APIRouter()

# 자동 감지할 위도/경도 컬럼명
_LAT_ALIASES = {"lat", "latitude", "위도", "y"}
_LNG_ALIASES = {"lng", "lon", "longitude", "경도", "x"}

MAX_ROWS = 50_000  # 안전 상한


def _detect_geo_cols(headers: list[str]) -> dict[str, str | None]:
    lower = {h: h.lower().strip() for h in headers}
    lat_col = next((h for h, l in lower.items() if l in _LAT_ALIASES), None)
    lng_col = next((h for h, l in lower.items() if l in _LNG_ALIASES), None)
    return {"latCol": lat_col, "lngCol": lng_col}


def _classify_headers(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric, categorical = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {"numericHeaders": numeric, "categoricalHeaders": categorical}


def _parse_df(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"파일이 너무 큽니다 ({len(df):,}행). 최대 {MAX_ROWS:,}행까지 지원합니다.",
        )

    # 컬럼명 앞뒤 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    headers = list(df.columns)
    geo = _detect_geo_cols(headers)
    col_types = _classify_headers(df)

    # NaN → None 처리 후 직렬화
    rows = to_native(df.where(pd.notnull(df), None).to_dict(orient="records"))

    return {
        "fileName": filename,
        "rowCount": len(rows),
        "headers": headers,
        **col_types,
        **geo,
        "rows": rows,
    }


# ──────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────

@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    CSV / TSV / Excel 파일을 파싱해서 rows + 메타데이터 반환.

    Response:
        fileName, rowCount, headers,
        numericHeaders, categoricalHeaders,
        latCol, lngCol (자동 감지, 없으면 null),
        rows: List[Dict]
    """
    name = file.filename or "unknown"
    content = await file.read()

    try:
        ext = name.rsplit(".", 1)[-1].lower()

        if ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl" if ext == "xlsx" else "xlrd")

        elif ext == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t", encoding="utf-8-sig")

        else:
            # CSV (기본) — 인코딩 자동 탐지
            for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(status_code=400, detail="파일 인코딩을 인식할 수 없습니다.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 파싱 오류: {str(e)}")

    return _parse_df(df, name)
