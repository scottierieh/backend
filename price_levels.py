"""
Reading a price level's label as an amount — shared by the conjoint routers
(conjoint_analysis.py, conjoint_hb.py, rc_analysis.py, aca_analysis.py,
acbc_analysis.py), which all need the number behind a price level to turn
utilities into willingness to pay.

Survey builders type price levels the way they would print them, not as bare
numbers: "12,000원", "₩12,000", "$1,299.99", "월 9,900원", "9,900원/월",
"1.2만원", "12,000 KRW". Each router used to strip the label with its own
one-liner, and they disagreed: four of them kept the thousands comma
("12,000"), so float() failed on every won price and WTP silently came back
empty; the CBC one dropped every non-digit, which read "1.2만원" as 1.2 and
glued the digits of "3개월 29,700원" into 329700.

Rules here:
  - thousands separators must be real groups of three ("12,000", not "12,00");
  - Korean magnitude units multiply the number they follow — 억 · 만 · 천 —
    and chain the way they are read aloud ("1만 2천원" = 12000,
    "12만9,900원" = 129900); a bare unit at the start counts as one
    ("만원" = 10000, "만5천원" = 15000); "k" right after a number is a
    thousand ("$1.2k"), but not the K of "KRW";
  - currency symbols, codes and period words around the amount are ignored;
  - "무료" / "free" / "공짜" are 0;
  - anything that is not exactly one amount — no number at all, or two
    separate numbers ("3개월 29,700원", "1~2만원") — is None, so callers skip
    that level instead of using a wrong number.

price_slope() is the other half the part-worth-based WTP routers (RC, ACA,
CBC-HB, ACBC) share: the least-squares slope of the price levels' utilities
against their amounts, i.e. utility per one unit of currency.

Stdlib only: the routers are imported at boot by main.py.
"""

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional

_UNIT = {'억': Decimal(100_000_000), '만': Decimal(10_000), '천': Decimal(1_000),
         'k': Decimal(1_000), 'K': Decimal(1_000)}

# One token: an optional sign, then either a number with an optional unit
# right after it, or a bare Korean unit that starts an amount. The sign and
# the bare unit only count when they do not sit inside a word ("A-1",
# "만족", "천연").
_TOKEN = re.compile(
    r'(?:(?<![0-9A-Za-z가-힣])(?P<sign>[-−]))?'
    r'(?:'
    r'(?P<num>\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)'
    r'\s*(?P<unit>억|만|천|[kK](?![A-Za-z]))?'
    r'|(?<![0-9A-Za-z가-힣])(?P<bare>억|만|천)(?=\d|원|\s|$)'
    r')'
)

_FREE = {'무료', 'free', '공짜'}


def parse_price_level(value) -> Optional[float]:
    """The amount a price level stands for, or None when it cannot be read."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    try:
        f = float(value)  # numpy scalars and the like
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        pass

    s = unicodedata.normalize('NFKC', str(value)).strip()
    if not s:
        return None
    if s.lower() in _FREE:
        return 0.0

    amounts = []          # finished amounts
    cur = None            # [total, magnitude of the last unit or None, sign, end]
    for m in _TOKEN.finditer(s):
        if m.group('bare'):
            num, mag = Decimal(1), _UNIT[m.group('bare')]
        else:
            try:
                num = Decimal(m.group('num').replace(',', ''))
            except InvalidOperation:
                return None
            mag = _UNIT.get(m.group('unit') or '')
        part = num * mag if mag is not None else num
        # Continue the amount in progress only for the next, smaller unit of
        # the same spoken number ("1만" then "2천"), separated by nothing but
        # spaces.
        if (cur is not None and cur[1] is not None and not m.group('sign')
                and (mag is None or mag < cur[1])
                and not s[cur[3]:m.start()].strip()):
            cur[0] += part
            cur[1] = mag
            cur[3] = m.end()
            continue
        if cur is not None:
            amounts.append(cur)
        cur = [part, mag, -1 if m.group('sign') else 1, m.end()]
    if cur is not None:
        amounts.append(cur)

    if len(amounts) != 1:
        return None
    total, _, sign, _ = amounts[0]
    f = float(sign * total)
    return f if math.isfinite(f) else None


def price_slope(levels) -> Optional[float]:
    """Utility per one unit of price, from (label, utility) pairs of the price levels.

    The least-squares slope of utility on amount (with an intercept) over the
    levels whose label reads as an amount; levels that do not read are left
    out. None when fewer than two distinct amounts read, or the slope is ~0
    (price does not move utility, so there is no ruler to convert with).
    """
    pts = []
    for label, utility in levels:
        x = parse_price_level(label)
        try:
            y = float(utility)
        except (TypeError, ValueError):
            continue
        if x is not None and math.isfinite(y):
            pts.append((x, y))
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    if not math.isfinite(slope) or abs(slope) < 1e-9:
        return None
    return slope
