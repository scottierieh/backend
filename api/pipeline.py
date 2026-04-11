"""
Data Pipeline API - Execution Engine
CRUD is managed by frontend (Firestore).
Backend handles: run, validate.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict, Literal
from datetime import datetime
import numpy as np
import pandas as pd
import time

router = APIRouter()


# ============================================
# REQUEST MODELS
# ============================================

class PipelineStepDef(BaseModel):
    type: str
    category: Literal['collect', 'clean', 'transform', 'feature', 'export']
    label: str
    description: Optional[str] = ''
    enabled: bool = True
    config: Dict[str, Any] = {}


class PipelineRunRequest(BaseModel):
    """Frontend sends steps + data for execution"""
    pipeline_id: Optional[str] = None
    steps: List[PipelineStepDef]
    data: Optional[List[Dict[str, Any]]] = None


class PipelineValidateRequest(BaseModel):
    steps: List[PipelineStepDef]


# ============================================
# HELPERS
# ============================================

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if pd.isna(obj): return None
    return obj


def _df_preview(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    rows = []
    for _, row in df.head(max_rows).iterrows():
        rows.append([_to_native(v) for v in row.values])
    return {
        "headers": list(df.columns),
        "rows": rows,
        "total_rows": len(df),
        "total_cols": len(df.columns),
    }


# ============================================
# STEP EXECUTION ENGINE
# ============================================

def _execute_step(step: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    stype = step["type"]
    config = step.get("config", {})

    # ── Collect (pass-through) ────────────────────────
    if stype in ('source_file', 'source_api', 'source_database'):
        if df is not None and len(df) > 0:
            return df
        raise ValueError(f"{stype}: No input data provided.")

    # ── Clean: Remove Duplicates ──────────────────────
    if stype == 'remove_duplicates':
        subset = config.get("subset", []) or None
        keep = config.get("keep", "first")
        if subset:
            subset = [c for c in subset if c in df.columns] or None
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    # ── Clean: Fill Missing ───────────────────────────
    if stype == 'fill_missing':
        columns = config.get("columns", [])
        method = config.get("method", "mean")
        targets = [c for c in columns if c in df.columns] if columns else list(df.columns)

        for col in targets:
            if df[col].isna().sum() == 0:
                continue
            if method in ('mean', 'median'):
                num = pd.to_numeric(df[col], errors='coerce')
                if num.notna().sum() > 0:
                    df[col] = df[col].fillna(num.mean() if method == 'mean' else num.median())
            elif method == 'mode':
                modes = df[col].mode()
                if len(modes) > 0:
                    df[col] = df[col].fillna(modes[0])
            elif method == 'zero':
                df[col] = df[col].fillna(0)
            elif method == 'forward':
                df[col] = df[col].ffill()
            elif method == 'backward':
                df[col] = df[col].bfill()
        return df

    # ── Clean: Remove Outliers ────────────────────────
    if stype == 'remove_outliers':
        columns = config.get("columns", [])
        method = config.get("method", "iqr")
        threshold = config.get("threshold", 1.5)
        targets = [c for c in columns if c in df.columns] if columns else list(df.select_dtypes(include=[np.number]).columns)

        mask = pd.Series(True, index=df.index)
        for col in targets:
            num = pd.to_numeric(df[col], errors='coerce')
            if method == 'iqr':
                q1, q3 = num.quantile(0.25), num.quantile(0.75)
                iqr = q3 - q1
                mask &= ((num >= q1 - threshold * iqr) & (num <= q3 + threshold * iqr)) | num.isna()
            elif method == 'zscore':
                mean, std = num.mean(), num.std()
                if std > 0:
                    mask &= (((num - mean) / std).abs() <= threshold) | num.isna()
        return df[mask].reset_index(drop=True)

    # ── Clean: Filter Rows ────────────────────────────
    if stype == 'filter_rows':
        column = config.get("column", "")
        operator = config.get("operator", "==")
        value = config.get("value", "")
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        col = df[column]
        if operator == 'not_null':
            return df[col.notna()].reset_index(drop=True)

        try:
            nval = float(value)
            ncol = pd.to_numeric(col, errors='coerce')
            ops = {'==': ncol == nval, '!=': ncol != nval, '>': ncol > nval,
                   '<': ncol < nval, '>=': ncol >= nval, '<=': ncol <= nval,
                   'contains': col.astype(str).str.contains(str(value), case=False, na=False)}
            mask = ops.get(operator, pd.Series(True, index=df.index))
        except (ValueError, TypeError):
            scol = col.astype(str)
            ops = {'==': scol == str(value), '!=': scol != str(value),
                   'contains': scol.str.contains(str(value), case=False, na=False)}
            mask = ops.get(operator, pd.Series(True, index=df.index))
        return df[mask].reset_index(drop=True)

    # ── Clean: Drop Columns ───────────────────────────
    if stype == 'drop_columns':
        cols = [c for c in config.get("columns", []) if c in df.columns]
        return df.drop(columns=cols) if cols else df

    # ── Transform: Type Cast ──────────────────────────
    if stype == 'type_cast':
        for col, dtype in config.get("mappings", {}).items():
            if col not in df.columns:
                continue
            try:
                if dtype in ('int', 'integer'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype in ('float', 'number'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype in ('str', 'text', 'string'):
                    df[col] = df[col].astype(str)
                elif dtype in ('date', 'datetime'):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'bool':
                    df[col] = df[col].astype(bool)
            except Exception:
                pass
        return df

    # ── Transform: Normalize ──────────────────────────
    if stype == 'normalize':
        method = config.get("method", "minmax")
        columns = config.get("columns", [])
        targets = [c for c in columns if c in df.columns] if columns else list(df.select_dtypes(include=[np.number]).columns)

        for col in targets:
            num = pd.to_numeric(df[col], errors='coerce')
            if method == 'minmax':
                mn, mx = num.min(), num.max()
                if mx - mn > 0: df[col] = (num - mn) / (mx - mn)
            elif method == 'zscore':
                mean, std = num.mean(), num.std()
                if std > 0: df[col] = (num - mean) / std
            elif method == 'robust':
                med = num.median()
                q1, q3 = num.quantile(0.25), num.quantile(0.75)
                if q3 - q1 > 0: df[col] = (num - med) / (q3 - q1)
        return df

    # ── Transform: Math Transform ─────────────────────
    if stype == 'math_transform':
        operation = config.get("operation", "log")
        columns = config.get("columns", [])
        targets = [c for c in columns if c in df.columns] if columns else list(df.select_dtypes(include=[np.number]).columns)

        for col in targets:
            num = pd.to_numeric(df[col], errors='coerce')
            if operation == 'log': df[col] = np.where(num > 0, np.log(num), np.nan)
            elif operation == 'log10': df[col] = np.where(num > 0, np.log10(num), np.nan)
            elif operation == 'sqrt': df[col] = np.where(num >= 0, np.sqrt(num), np.nan)
            elif operation == 'square': df[col] = num ** 2
            elif operation == 'abs': df[col] = num.abs()
            elif operation == 'round': df[col] = num.round()
        return df

    # ── Transform: Rename Columns ─────────────────────
    if stype == 'rename_columns':
        mappings = {k: v for k, v in config.get("mappings", {}).items() if k in df.columns}
        return df.rename(columns=mappings) if mappings else df

    # ── Transform: Sort ───────────────────────────────
    if stype == 'sort':
        column = config.get("column", "")
        direction = config.get("direction", "asc")
        if column in df.columns:
            return df.sort_values(by=column, ascending=(direction == 'asc')).reset_index(drop=True)
        return df

    # ── Feature: One-Hot Encoding ─────────────────────
    if stype == 'one_hot_encoding':
        columns = config.get("columns", [])
        drop_first = config.get("dropFirst", False)
        prefix = config.get("prefix", "")
        targets = [c for c in columns if c in df.columns] if columns else []

        for col in targets:
            p = prefix if prefix else col
            dummies = pd.get_dummies(df[col], prefix=p, drop_first=drop_first, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    # ── Feature: Date Features ────────────────────────
    if stype == 'date_features':
        column = config.get("column", "")
        features = config.get("features", ["year", "month", "day", "dayOfWeek"])
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        dt = pd.to_datetime(df[column], errors='coerce')
        fmap = {
            'year': dt.dt.year, 'month': dt.dt.month, 'day': dt.dt.day,
            'dayOfWeek': dt.dt.dayofweek, 'hour': dt.dt.hour,
            'quarter': dt.dt.quarter, 'weekOfYear': dt.dt.isocalendar().week.astype(int),
        }
        for f in features:
            if f in fmap:
                df[f"{column}_{f}"] = fmap[f]
        return df

    # ── Feature: Binning ──────────────────────────────
    if stype == 'binning':
        column = config.get("column", "")
        bins = config.get("bins", 5)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        df[f"{column}_bin"] = pd.cut(pd.to_numeric(df[column], errors='coerce'), bins=bins, labels=False)
        return df

    # ── Feature: Custom Formula ───────────────────────
    if stype == 'custom_formula':
        new_col = config.get("newColumn", "")
        formula = config.get("formula", "")
        if not new_col or not formula:
            raise ValueError("Both newColumn and formula are required")
        try:
            df[new_col] = df.eval(formula)
        except Exception as e:
            raise ValueError(f"Formula error: {str(e)}")
        return df

    # ── Export (pass-through) ─────────────────────────
    if stype in ('export_file', 'export_database'):
        return df

    return df


# ============================================
# ENDPOINTS
# ============================================

@router.post("/run")
def run_pipeline(req: PipelineRunRequest):
    """Execute pipeline steps on provided data."""
    if not req.steps:
        raise HTTPException(status_code=400, detail="No steps provided")

    df = pd.DataFrame(req.data) if req.data else pd.DataFrame()

    started_at = datetime.utcnow().isoformat() + 'Z'
    total_start = time.time()
    step_results = []
    succeeded = failed = skipped = 0
    final_status = "completed"

    for i, step_def in enumerate(req.steps):
        step = step_def.dict()

        if not step.get("enabled", True):
            step_results.append({
                "step_index": i, "step_label": step.get("label", f"Step {i+1}"),
                "status": "skipped", "duration_ms": 0,
                "rows_in": len(df), "rows_out": len(df),
                "cols_in": len(df.columns) if not df.empty else 0,
                "cols_out": len(df.columns) if not df.empty else 0,
                "preview": None, "error": None,
            })
            skipped += 1
            continue

        rows_in = len(df)
        cols_in = len(df.columns) if not df.empty else 0
        t0 = time.time()

        try:
            df = _execute_step(step, df)
            ms = int((time.time() - t0) * 1000)
            step_results.append({
                "step_index": i, "step_label": step.get("label", f"Step {i+1}"),
                "status": "success", "duration_ms": ms,
                "rows_in": rows_in, "rows_out": len(df),
                "cols_in": cols_in, "cols_out": len(df.columns),
                "preview": _df_preview(df, max_rows=5), "error": None,
            })
            succeeded += 1
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            step_results.append({
                "step_index": i, "step_label": step.get("label", f"Step {i+1}"),
                "status": "error", "duration_ms": ms,
                "rows_in": rows_in, "rows_out": rows_in,
                "cols_in": cols_in, "cols_out": cols_in,
                "preview": None, "error": str(e),
            })
            failed += 1
            final_status = "failed"
            break

    total_ms = int((time.time() - total_start) * 1000)
    finished_at = datetime.utcnow().isoformat() + 'Z'

    output_preview = None
    if not df.empty and final_status == "completed":
        output_preview = _df_preview(df, max_rows=10)

    return {
        "success": final_status == "completed",
        "pipeline_id": req.pipeline_id,
        "status": final_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_duration_ms": total_ms,
        "steps_total": len(req.steps),
        "steps_succeeded": succeeded,
        "steps_failed": failed,
        "steps_skipped": skipped,
        "step_results": step_results,
        "output_preview": output_preview,
    }


@router.post("/validate")
def validate_pipeline(req: PipelineValidateRequest):
    """Validate pipeline steps without running."""
    issues = []
    valid_types = {
        'collect': ['source_file', 'source_api', 'source_database'],
        'clean': ['remove_duplicates', 'fill_missing', 'remove_outliers', 'filter_rows', 'drop_columns'],
        'transform': ['type_cast', 'normalize', 'math_transform', 'rename_columns', 'sort'],
        'feature': ['one_hot_encoding', 'date_features', 'binning', 'custom_formula'],
        'export': ['export_file', 'export_database'],
    }

    for i, step in enumerate(req.steps):
        s = step.dict()
        cat, stype, config = s["category"], s["type"], s.get("config", {})

        if cat not in valid_types:
            issues.append({"step": i, "level": "error", "message": f"Unknown category: {cat}"})
            continue
        if stype not in valid_types[cat]:
            issues.append({"step": i, "level": "warning", "message": f"Type '{stype}' not standard for '{cat}'"})

        if stype == 'filter_rows' and not config.get("column"):
            issues.append({"step": i, "level": "error", "message": "Filter Rows requires 'column'"})
        if stype == 'sort' and not config.get("column"):
            issues.append({"step": i, "level": "error", "message": "Sort requires 'column'"})
        if stype == 'custom_formula':
            if not config.get("newColumn"):
                issues.append({"step": i, "level": "error", "message": "Custom Formula requires 'newColumn'"})
            if not config.get("formula"):
                issues.append({"step": i, "level": "error", "message": "Custom Formula requires 'formula'"})
        if stype in ('date_features', 'binning') and not config.get("column"):
            issues.append({"step": i, "level": "error", "message": f"{s.get('label', stype)} requires 'column'"})

    has_collect = any(s.enabled and s.category == 'collect' for s in req.steps)
    if not has_collect:
        issues.append({"step": -1, "level": "warning", "message": "No enabled collect step. Data must be provided at runtime."})

    errors = [x for x in issues if x["level"] == "error"]
    return {
        "success": True,
        "valid": len(errors) == 0,
        "errors": len(errors),
        "warnings": len(issues) - len(errors),
        "issues": issues,
    }
