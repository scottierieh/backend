"""
Statistica Scheduler — APScheduler 기반 자동 데이터 수집
Firebase Admin SDK로 Firestore에 직접 저장
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

# ─── Firebase Admin Init ──────────────────────────────────────────────────────

_db: Any = None

def get_db():
    global _db
    if _db is None:
        if not firebase_admin._apps:
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
            else:
                cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def date_str() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


TEMP_ORG_ID = os.getenv('DEFAULT_ORG_ID', 'default_org')

# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_file_to_firestore(db, file_id: str, file_name: str, csv: str,
                            data_type: str, description: str,
                            columns: list, column_types: list,
                            source_platform: str, org_id: str = TEMP_ORG_ID):
    db.collection('shared-files').document(file_id).set({
        'fileName':       file_name,
        'fileSize':       len(csv),
        'fileType':       '.csv',
        'orgId':          org_id,
        'uploadedBy':     'system_scheduler',
        'uploadedByEmail':'scheduler@statistica.ai',
        'description':    description,
        'createdAt':      firestore.SERVER_TIMESTAMP,
        'downloadURL':    'data:text/csv;charset=utf-8,' + csv,
        'autoMapped':     True,
        'dataType':       data_type,
        'columns':        columns,
        'columnTypes':    column_types,
        'sourcePlatform': source_platform,
        'syncedAt':       now_iso(),
        'scheduledSync':  True,
    })


def update_schedule_status(db, job_id: str, status: str,
                            error: str = None, org_id: str = TEMP_ORG_ID):
    data = {
        'lastSyncAt':     firestore.SERVER_TIMESTAMP,
        'lastSyncStatus': status,
    }
    if error:
        data['lastSyncError'] = error
    db.collection('orgs').document(org_id)\
      .collection('schedules').document(job_id)\
      .set(data, merge=True)


def get_schedule_config(db, job_id: str, org_id: str = TEMP_ORG_ID) -> dict | None:
    doc = db.collection('orgs').document(org_id)\
            .collection('schedules').document(job_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    if not data.get('enabled', False):
        return None
    return data.get('config', {})

# ─── Yahoo Finance sync ───────────────────────────────────────────────────────

async def job_market_data():
    import aiohttp
    db = get_db()
    config = get_schedule_config(db, 'market_data')
    if config is None:
        return

    tickers      = config.get('tickers', [])
    period       = config.get('period', '1mo')
    analysis     = config.get('analysisTypes', ['basic_stats'])

    if not tickers:
        return

    logger.info(f'[MarketData] Starting sync for {tickers}')

    try:
        # Get Yahoo crumb
        async with aiohttp.ClientSession() as session:
            async with session.get('https://fc.yahoo.com',
                                   headers={'User-Agent': 'Mozilla/5.0'},
                                   allow_redirects=False) as r:
                cookies = '; '.join(
                    f"{c.key}={c.value}" for c in r.cookies.values()
                )

            async with session.get(
                'https://query2.finance.yahoo.com/v1/test/getcrumb',
                headers={'User-Agent': 'Mozilla/5.0', 'Cookie': cookies}
            ) as r:
                crumb = await r.text()

            price_data = {}
            for ticker in tickers:
                try:
                    url = (f'https://query2.finance.yahoo.com/v8/finance/chart/'
                           f'{ticker}?range={period}&interval=1d'
                           f'&crumb={crumb}')
                    async with session.get(
                        url, headers={'User-Agent': 'Mozilla/5.0', 'Cookie': cookies}
                    ) as r:
                        data = await r.json()
                    result = data.get('chart', {}).get('result', [None])[0]
                    if not result:
                        continue
                    timestamps = result.get('timestamp', [])
                    quote      = result.get('indicators', {}).get('quote', [{}])[0]
                    adj_close  = (result.get('indicators', {})
                                  .get('adjclose', [{}])[0]
                                  .get('adjclose', quote.get('close', [])))
                    price_data[ticker] = {
                        'dates': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                                  for ts in timestamps],
                        'close': adj_close,
                        'volume': quote.get('volume', []),
                    }
                except Exception as e:
                    logger.warning(f'[MarketData] {ticker} failed: {e}')

        if 'basic_stats' in analysis and price_data:
            rows = ['ticker,date,close,daily_return']
            for ticker, pd_ in price_data.items():
                for i, (d, c) in enumerate(zip(pd_['dates'], pd_['close'])):
                    if c is None:
                        continue
                    ret = 0.0
                    if i > 0 and pd_['close'][i-1]:
                        ret = (c - pd_['close'][i-1]) / pd_['close'][i-1] * 100
                    rows.append(f"{ticker},{d},{c:.2f},{ret:.4f}")

            csv = '\n'.join(rows)
            file_id = f"scheduled_market_{TEMP_ORG_ID}_{int(datetime.now().timestamp())}"
            save_file_to_firestore(
                db, file_id,
                f'market_data_{date_str()}.csv', csv,
                'basic_stats',
                f"Scheduled market data — {', '.join(tickers)} ({period})",
                ['ticker', 'date', 'close', 'daily_return'],
                ['categorical', 'datetime', 'numeric', 'numeric'],
                'yahoo_finance',
            )
            logger.info(f'[MarketData] Saved {len(rows)-1} rows')

        update_schedule_status(db, 'market_data', 'success')

    except Exception as e:
        logger.error(f'[MarketData] Error: {e}')
        update_schedule_status(db, 'market_data', 'error', str(e))


# ─── FRED sync ────────────────────────────────────────────────────────────────

async def job_macro_data():
    import aiohttp
    db = get_db()
    config = get_schedule_config(db, 'macro_data')
    if config is None:
        return

    fred_api_key  = os.getenv('FRED_API_KEY', config.get('fredApiKey', ''))
    bls_api_key   = os.getenv('BLS_API_KEY',  config.get('blsApiKey',  ''))
    fred_series   = config.get('fredSeriesIds', [])
    bls_series    = config.get('blsSeriesIds',  [])

    logger.info(f'[MacroData] FRED={fred_series} BLS={bls_series}')

    try:
        async with aiohttp.ClientSession() as session:

            # ── FRED ──
            if fred_api_key and fred_series:
                series_data = []
                for sid in fred_series:
                    params = {
                        'series_id': sid, 'api_key': fred_api_key,
                        'file_type': 'json', 'sort_order': 'asc',
                        'observation_start': f'{datetime.now().year - 1}-01-01',
                    }
                    async with session.get(
                        'https://api.stlouisfed.org/fred/series/observations',
                        params=params
                    ) as r:
                        data = await r.json()
                    obs = [o for o in data.get('observations', []) if o['value'] != '.']
                    series_data.append({'id': sid, 'obs': obs})

                all_dates = sorted(set(
                    o['date'] for s in series_data for o in s['obs']
                ))
                rows = [','.join(['date'] + [s['id'] for s in series_data])]
                for date in all_dates:
                    vals = [next((o['value'] for o in s['obs'] if o['date'] == date), '')
                            for s in series_data]
                    rows.append(','.join([date] + vals))

                csv = '\n'.join(rows)
                file_id = f"scheduled_fred_{TEMP_ORG_ID}_{int(datetime.now().timestamp())}"
                save_file_to_firestore(
                    db, file_id,
                    f'fred_macro_{date_str()}.csv', csv,
                    'fred_macro',
                    f"Scheduled FRED sync — {', '.join(fred_series)}",
                    ['date'] + fred_series,
                    ['datetime'] + ['numeric'] * len(fred_series),
                    'macro_api',
                )
                logger.info(f'[MacroData] FRED saved {len(rows)-1} rows')

            # ── BLS ──
            if bls_api_key and bls_series:
                year = datetime.now().year
                payload = {
                    'seriesid':        bls_series,
                    'startyear':       str(year - 1),
                    'endyear':         str(year),
                    'registrationkey': bls_api_key,
                }
                async with session.post(
                    'https://api.bls.gov/publicAPI/v2/timeseries/data/',
                    json=payload
                ) as r:
                    data = await r.json()

                bls_data = {}
                for s in data.get('Results', {}).get('series', []):
                    bls_data[s['seriesID']] = sorted(
                        [{'date': f"{d['year']}-{d['period'].replace('M','').zfill(2)}-01",
                          'value': d['value']}
                         for d in s.get('data', []) if d['period'] != 'M13'],
                        key=lambda x: x['date']
                    )

                all_dates = sorted(set(
                    d['date'] for arr in bls_data.values() for d in arr
                ))
                rows = [','.join(['date'] + bls_series)]
                for date in all_dates:
                    vals = [next((d['value'] for d in bls_data.get(sid, [])
                                  if d['date'] == date), '')
                            for sid in bls_series]
                    rows.append(','.join([date] + vals))

                csv = '\n'.join(rows)
                file_id = f"scheduled_bls_{TEMP_ORG_ID}_{int(datetime.now().timestamp())}"
                save_file_to_firestore(
                    db, file_id,
                    f'bls_macro_{date_str()}.csv', csv,
                    'bls_macro',
                    f"Scheduled BLS sync — {', '.join(bls_series)}",
                    ['date'] + bls_series,
                    ['datetime'] + ['numeric'] * len(bls_series),
                    'macro_api',
                )
                logger.info(f'[MacroData] BLS saved {len(rows)-1} rows')

        update_schedule_status(db, 'macro_data', 'success')

    except Exception as e:
        logger.error(f'[MacroData] Error: {e}')
        update_schedule_status(db, 'macro_data', 'error', str(e))


# ─── Korean Stats sync ────────────────────────────────────────────────────────

async def job_korea_stats():
    import aiohttp
    db = get_db()
    config = get_schedule_config(db, 'korea_stats')
    if config is None:
        return

    kosis_api_key = os.getenv('KOSIS_API_KEY', '')
    ecos_api_key  = os.getenv('ECOS_API_KEY',  '')
    kosis_items   = config.get('kosisItems', [])
    ecos_items    = config.get('ecosItems',  [])

    logger.info(f'[KoreaStats] KOSIS={len(kosis_items)} ECOS={len(ecos_items)}')

    try:
        async with aiohttp.ClientSession() as session:

            # ── KOSIS ──
            if kosis_api_key and kosis_items:
                for item in kosis_items:
                    params = {
                        'method': 'getList', 'apiKey': kosis_api_key,
                        'format': 'json', 'jsonVD': 'Y',
                        'orgId': item['orgId'], 'tblId': item['tblId'],
                        'itmId': item.get('itmId', 'ALL'),
                        'objL1': item.get('objL1', 'ALL'),
                        'prdSe': item.get('prdSe', 'Y'),
                        'newEstPrdCnt': '12',
                    }
                    async with session.get(
                        'https://kosis.kr/openapi/statisticsData.do',
                        params=params
                    ) as r:
                        data = await r.json()

                    if not isinstance(data, list):
                        continue

                    rows = ['period,item,value,unit']
                    for row in data:
                        rows.append(
                            f"{row.get('PRD_DE','')},\"{row.get('ITM_NM','')}\","
                            f"{row.get('DT','')},\"{row.get('UNIT_NM','')}\""
                        )

                    csv = '\n'.join(rows)
                    label = item.get('label', item['tblId'])
                    file_id = f"scheduled_kosis_{item['tblId']}_{TEMP_ORG_ID}_{int(datetime.now().timestamp())}"
                    save_file_to_firestore(
                        db, file_id,
                        f"kosis_{label}_{date_str()}.csv", csv,
                        f"kosis_{item['tblId']}",
                        f"Scheduled KOSIS — {label} — 출처: KOSIS(국가데이터처)",
                        ['period', 'item', 'value', 'unit'],
                        ['datetime', 'categorical', 'numeric', 'categorical'],
                        'korea_stats',
                    )
                    logger.info(f'[KoreaStats] KOSIS {label} saved {len(rows)-1} rows')

            # ── ECOS ──
            if ecos_api_key and ecos_items:
                year = datetime.now().year
                for item in ecos_items:
                    cycle = item.get('cycle', 'M')
                    start = str(year - 1) if cycle == 'A' else f"{year - 1}01"
                    end   = str(year)     if cycle == 'A' else f"{year}12"
                    url   = (f"https://ecos.bok.or.kr/api/StatisticSearch/"
                             f"{ecos_api_key}/json/kr/1/10000/"
                             f"{item['statCode']}/{cycle}/{start}/{end}/*/*/*")

                    async with session.get(url) as r:
                        data = await r.json()

                    ecos_rows = data.get('StatisticSearch', {}).get('row', [])
                    if not ecos_rows:
                        continue

                    rows = ['period,item,value,unit']
                    for row in ecos_rows:
                        rows.append(
                            f"{row.get('TIME','')},\"{row.get('ITEM_NAME1','')}\","
                            f"{row.get('DATA_VALUE','')},\"{row.get('UNIT_NAME','')}\""
                        )

                    csv = '\n'.join(rows)
                    label = item.get('label', item['statCode'])
                    file_id = f"scheduled_ecos_{item['statCode']}_{TEMP_ORG_ID}_{int(datetime.now().timestamp())}"
                    save_file_to_firestore(
                        db, file_id,
                        f"ecos_{label}_{date_str()}.csv", csv,
                        f"ecos_{item['statCode']}",
                        f"Scheduled ECOS — {label} — 출처: 한국은행 ECOS",
                        ['period', 'item', 'value', 'unit'],
                        ['datetime', 'categorical', 'numeric', 'categorical'],
                        'korea_stats',
                    )
                    logger.info(f'[KoreaStats] ECOS {label} saved {len(rows)-1} rows')

        update_schedule_status(db, 'korea_stats', 'success')

    except Exception as e:
        logger.error(f'[KoreaStats] Error: {e}')
        update_schedule_status(db, 'korea_stats', 'error', str(e))


# ─── Manual trigger endpoint helper ───────────────────────────────────────────

JOB_FUNCTIONS = {
    'market_data': job_market_data,
    'macro_data':  job_macro_data,
    'korea_stats': job_korea_stats,
}

async def run_job_now(job_id: str) -> dict:
    fn = JOB_FUNCTIONS.get(job_id)
    if not fn:
        return {'success': False, 'error': f'Unknown job: {job_id}'}
    try:
        await fn()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ─── Scheduler setup ──────────────────────────────────────────────────────────

def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone='UTC')

    scheduler.add_job(
        job_market_data, CronTrigger(hour=1, minute=0),
        id='market_data', name='Market Data Sync',
        replace_existing=True, misfire_grace_time=3600,
    )
    scheduler.add_job(
        job_macro_data, CronTrigger(hour=2, minute=0),
        id='macro_data', name='Macro Data Sync',
        replace_existing=True, misfire_grace_time=3600,
    )
    scheduler.add_job(
        job_korea_stats, CronTrigger(hour=3, minute=0),
        id='korea_stats', name='Korean Statistics Sync',
        replace_existing=True, misfire_grace_time=3600,
    )

    return scheduler
