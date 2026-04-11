"""
schemas.py — Shared Pydantic models used across all routers.
Single source of truth: no more duplicated AssetIn across 9 files.
"""

from typing import List, Optional
from pydantic import BaseModel


class AssetIn(BaseModel):
    ticker:           str
    returns:          List[float]
    weight:           float = 1.0
    prices:           Optional[List[float]] = None
    periods:          Optional[List[str]]   = None
    # ADV / liquidity
    adv:              Optional[float] = None   # average daily volume USD
    bidAsk:           Optional[float] = None   # bid-ask spread bps
    # Identity
    sector:           Optional[str]   = None   # required for sector-relative analysis
    # Fundamentals — Value
    pe:               Optional[float] = None
    pb:               Optional[float] = None
    evEbitda:         Optional[float] = None
    divYield:         Optional[float] = None
    fcfYield:         Optional[float] = None
    marketCap:        Optional[float] = None
    # Fundamentals — Quality
    roe:              Optional[float] = None
    roa:              Optional[float] = None
    grossMargin:      Optional[float] = None
    operatingMargin:  Optional[float] = None
    netMargin:        Optional[float] = None
    revenueGrowth:    Optional[float] = None
    debtEquity:       Optional[float] = None
    interestCoverage: Optional[float] = None


class BenchmarkIn(BaseModel):
    name:    str
    returns: List[float]
    periods: Optional[List[str]] = None
