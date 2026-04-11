'use client';

import React, { useMemo, useState } from 'react';
import { Upload, CheckCircle2, AlertCircle, ChevronRight, RefreshCw, X, Database, Plus, ChevronDown, ChevronUp, Info, FileSpreadsheet, Target } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { type InvestmentPageProps, calcStats } from '../../investment-intelligence-app';

function fmt(n: number) {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (Math.abs(n) >= 1_000)     return (n / 1_000).toFixed(2) + 'K';
  return n.toFixed(2);
}
function pctFmt(n: number) { return (n >= 0 ? '+' : '') + (n * 100).toFixed(2) + '%'; }

// ── Fundamental fields definition ─────────────────────────────────────────────
const FUNDAMENTAL_FIELDS = [
  // Value
  { key: 'pe',              label: 'P/E Ratio',       hint: 'Price / Earnings',             unit: 'x',  placeholder: '15.0' },
  { key: 'pb',              label: 'P/B Ratio',        hint: 'Price / Book Value',           unit: 'x',  placeholder: '2.0'  },
  { key: 'evEbitda',        label: 'EV/EBITDA',        hint: 'Enterprise Value / EBITDA',    unit: 'x',  placeholder: '10.0' },
  { key: 'divYield',        label: 'Div Yield',        hint: 'Annual dividend yield',        unit: '%',  placeholder: '2.5'  },
  { key: 'fcfYield',        label: 'FCF Yield',        hint: 'Free cash flow yield',         unit: '%',  placeholder: '4.0'  },
  // Quality — profitability
  { key: 'roe',             label: 'ROE',              hint: 'Return on Equity',             unit: '%',  placeholder: '18.0' },
  { key: 'roa',             label: 'ROA',              hint: 'Return on Assets',             unit: '%',  placeholder: '8.0'  },
  { key: 'grossMargin',     label: 'Gross Margin',     hint: 'Gross profit margin',          unit: '%',  placeholder: '45.0' },
  { key: 'operatingMargin', label: 'Op. Margin',       hint: 'Operating profit margin',      unit: '%',  placeholder: '20.0' },
  { key: 'netMargin',       label: 'Net Margin',       hint: 'Net profit margin',            unit: '%',  placeholder: '15.0' },
  { key: 'revenueGrowth',   label: 'Rev Growth',       hint: 'YoY revenue growth',           unit: '%',  placeholder: '12.0' },
  // Quality — leverage
  { key: 'debtEquity',      label: 'D/E Ratio',        hint: 'Total Debt / Equity',          unit: 'x',  placeholder: '0.5'  },
  { key: 'interestCoverage',label: 'Int. Coverage',    hint: 'EBIT / Interest Expense',      unit: 'x',  placeholder: '10.0' },
] as const;

type FundamentalKey = typeof FUNDAMENTAL_FIELDS[number]['key'];

// ── Sample datasets — with fundamentals ──────────────────────────────────────
const SAMPLES = [
  {
    id: 'us-tech',
    label: 'US Tech Portfolio',
    desc: '5 mega-cap tech stocks · 24 months',
    color: '#6C3AED',
    meta: 'AAPL · MSFT · GOOGL · AMZN · NVDA',
    csv: `Date,AAPL,MSFT,GOOGL,AMZN,NVDA
2023-01-31,141.9,239.2,94.0,103.4,143.2
2023-02-28,147.9,252.8,95.4,103.2,175.4
2023-03-31,160.8,288.0,104.5,102.6,277.8
2023-04-30,169.7,307.3,107.1,106.1,277.5
2023-05-31,177.8,328.4,124.1,119.3,402.3
2023-06-30,193.9,340.5,124.7,130.4,423.0
2023-07-31,178.2,335.9,127.0,133.7,467.7
2023-08-31,187.6,328.0,129.0,138.2,493.5
2023-09-30,171.2,315.8,130.3,127.1,434.0
2023-10-31,173.0,329.0,139.5,133.9,405.5
2023-11-30,189.7,374.5,140.8,153.9,497.0
2023-12-31,192.5,374.0,140.9,153.4,495.2
2024-01-31,186.0,397.0,155.4,171.8,613.1
2024-02-29,184.4,415.0,162.6,179.6,788.2
2024-03-31,171.2,420.5,160.6,181.2,903.6
2024-04-30,169.9,406.3,161.5,180.0,762.0
2024-05-31,192.4,430.2,178.0,183.2,1064.7
2024-06-30,213.3,445.9,185.3,194.5,1208.9
2024-07-31,218.8,448.2,180.2,186.1,1179.6
2024-08-31,226.5,410.6,170.1,193.5,1237.0
2024-09-30,233.0,441.5,164.0,196.1,1202.8
2024-10-31,225.9,430.0,168.7,195.9,1404.6
2024-11-30,237.4,417.0,180.8,215.4,1321.7
2024-12-31,248.5,447.0,195.4,224.0,1376.5
## FUNDAMENTALS
Ticker,Sector,PE,PB,EV_EBITDA,DIV_YIELD,FCF_YIELD,ROE,ROA,GROSS_MARGIN,OPERATING_MARGIN,NET_MARGIN,REV_GROWTH,DEBT_EQUITY,INTEREST_COVERAGE
AAPL,Technology,28.5,45.2,22.1,0.5,3.8,160.1,28.3,43.3,29.2,23.9,2.8,1.5,28.4
MSFT,Technology,35.2,12.8,25.4,0.7,3.2,39.2,18.6,69.4,44.6,34.1,15.7,0.4,52.1
GOOGL,Communication Services,24.1,6.8,18.6,0.0,4.1,27.3,14.8,56.9,26.5,23.7,8.7,0.1,38.2
AMZN,Consumer Discretionary,42.8,8.9,32.1,0.0,2.9,19.8,6.2,47.6,9.8,7.4,12.3,0.7,12.8
NVDA,Technology,65.4,32.1,48.2,0.0,2.1,91.2,45.2,72.7,54.1,48.9,122.4,0.4,84.6,3300000000000`,
  },
  {
    id: 'multi-asset',
    label: 'Multi-Asset Portfolio',
    desc: 'Stocks + bonds + commodities · 24 months',
    color: '#10B981',
    meta: 'SPY · TLT · GLD · QQQ · VNQ',
    csv: `Date,SPY,TLT,GLD,QQQ,VNQ
2023-01-31,386.2,106.2,179.8,285.4,85.4
2023-02-28,389.9,99.8,178.5,293.4,82.6
2023-03-31,404.1,105.2,186.7,318.4,82.4
2023-04-30,412.1,103.6,193.0,317.3,81.8
2023-05-31,413.8,97.2,183.4,324.1,76.3
2023-06-30,436.7,94.2,185.7,364.6,78.5
2023-07-31,450.1,93.2,188.3,380.3,78.1
2023-08-31,440.1,88.7,186.7,369.5,73.9
2023-09-30,423.0,82.2,181.0,353.3,68.9
2023-10-31,407.2,78.5,182.2,345.5,64.6
2023-11-30,444.5,87.3,188.5,388.7,72.4
2023-12-31,458.3,93.5,192.8,404.0,76.2
2024-01-31,468.6,92.8,203.4,424.4,75.1
2024-02-29,496.0,89.7,203.8,448.9,74.2
2024-03-31,519.3,91.4,215.1,447.3,78.3
2024-04-30,499.6,84.2,220.2,427.3,72.4
2024-05-31,527.4,86.3,228.6,460.8,74.8
2024-06-30,544.2,86.7,231.8,478.3,75.3
2024-07-31,549.6,96.3,237.5,483.6,80.2
2024-08-31,555.0,100.3,244.8,476.5,85.5
2024-09-30,565.4,98.1,253.4,483.0,85.9
2024-10-31,576.9,89.4,249.8,492.0,80.5
2024-11-30,591.5,88.7,261.3,510.2,83.0
2024-12-31,585.7,85.6,257.2,502.4,80.8
## FUNDAMENTALS
Ticker,Sector,PE,PB,EV_EBITDA,DIV_YIELD,FCF_YIELD,ROE,ROA,GROSS_MARGIN,OPERATING_MARGIN,NET_MARGIN,REV_GROWTH,DEBT_EQUITY,INTEREST_COVERAGE
SPY,Broad Market,22.4,4.1,16.8,1.3,3.6,18.2,7.8,35.4,19.2,16.8,7.2,0.9,14.2
TLT,Fixed Income,,,,,4.2,,,,,,-0.8,
GLD,Commodities,,,,,2.1,,100.0,,,,-1.2,
QQQ,Technology,30.2,6.8,22.4,0.6,3.1,24.8,11.2,52.1,28.4,22.6,12.4,0.6,24.8
VNQ,Real Estate,38.1,2.2,24.6,4.1,2.8,8.4,3.2,68.2,42.1,31.4,4.1,1.8,3.8,32000000000`,
  },
  {
    id: 'asia-em',
    label: 'Asia EM Portfolio',
    desc: 'Korean & Taiwanese tech ETFs · 24 months',
    color: '#F59E0B',
    meta: 'SAMSUNG · TSMC · HYNIX · LG · MEDIATEK',
    csv: `Date,SAMSUNG,TSMC,HYNIX,LG,MEDIATEK
2023-01-31,58400,85.2,75200,98000,712
2023-02-28,61200,93.4,83000,102000,748
2023-03-31,65800,100.5,87000,107000,810
2023-04-30,67200,98.4,85000,112000,795
2023-05-31,70000,102.3,90000,108000,834
2023-06-30,71500,105.8,92000,115000,862
2023-07-31,69000,101.2,88000,112000,840
2023-08-31,67500,98.6,85000,109000,812
2023-09-30,65000,96.2,82000,106000,784
2023-10-31,68000,99.8,86000,110000,820
2023-11-30,72000,106.4,92000,117000,875
2023-12-31,74500,110.2,95000,121000,912
2024-01-31,76000,115.4,97000,124000,948
2024-02-29,78000,122.5,102000,128000,1024
2024-03-31,80000,130.8,108000,132000,1102
2024-04-30,77000,126.4,104000,128000,1056
2024-05-31,80500,132.6,110000,134000,1120
2024-06-30,82000,138.2,115000,138000,1185
2024-07-31,84000,142.5,118000,141000,1242
2024-08-31,81000,138.8,114000,138000,1208
2024-09-30,79000,135.2,112000,136000,1172
2024-10-31,82000,140.6,116000,140000,1234
2024-11-30,85000,148.3,122000,145000,1312
2024-12-31,88000,155.6,128000,150000,1398
## FUNDAMENTALS
Ticker,Sector,PE,PB,EV_EBITDA,DIV_YIELD,FCF_YIELD,ROE,ROA,GROSS_MARGIN,OPERATING_MARGIN,NET_MARGIN,REV_GROWTH,DEBT_EQUITY,INTEREST_COVERAGE
SAMSUNG,Technology,14.2,1.5,8.4,2.1,5.2,11.2,5.8,38.4,14.2,10.8,8.2,0.3,24.6
TSMC,Semiconductors,22.8,6.2,14.6,1.8,3.8,28.4,16.4,53.1,38.2,31.8,24.6,0.2,42.8
HYNIX,Semiconductors,18.4,1.8,9.2,1.2,4.1,14.8,6.2,32.6,18.4,12.6,35.4,0.6,18.2
LG,Consumer Electronics,11.4,0.9,7.8,3.2,6.1,8.2,3.4,24.1,7.8,5.2,3.8,0.8,8.4
MEDIATEK,Semiconductors,16.8,3.4,11.2,4.8,5.8,22.4,12.8,46.8,22.4,18.6,18.2,0.2,32.4,52000000000`,
  },
];

// ── Sample fundamentals CSVs (separate from price data) ──────────────────────
const SAMPLE_FUNDAMENTALS: Record<string, string> = {
  'us-tech': `ticker,sector,pe,pb,ev_ebitda,div_yield,fcf_yield,roe,roa,gross_margin,operating_margin,net_margin,rev_growth,debt_equity,interest_coverage
AAPL,Technology,28.5,45.2,22.1,0.5,3.8,160.1,28.3,43.3,29.2,23.9,2.8,1.5,28.4
MSFT,Technology,35.2,12.8,25.4,0.7,3.2,39.2,18.6,69.4,44.6,34.1,15.7,0.4,52.1
GOOGL,Communication Services,24.1,6.8,18.6,0.0,4.1,27.3,14.8,56.9,26.5,23.7,8.7,0.1,38.2
AMZN,Consumer Discretionary,42.8,8.9,32.1,0.0,2.9,19.8,6.2,47.6,9.8,7.4,12.3,0.7,12.8
NVDA,Technology,65.4,32.1,48.2,0.0,2.1,91.2,45.2,72.7,54.1,48.9,122.4,0.4,84.6,3300000000000`,
  'multi-asset': `ticker,sector,pe,pb,ev_ebitda,div_yield,fcf_yield,roe,roa,gross_margin,operating_margin,net_margin,rev_growth,debt_equity,interest_coverage
SPY,Broad Market,22.4,4.1,16.8,1.3,3.6,18.2,7.8,35.4,19.2,16.8,7.2,0.9,14.2
TLT,Fixed Income,,,,,4.2,,,,,,-0.8,
GLD,Commodities,,,,,2.1,,100.0,,,,-1.2,
QQQ,Technology,30.2,6.8,22.4,0.6,3.1,24.8,11.2,52.1,28.4,22.6,12.4,0.6,24.8
VNQ,Real Estate,38.1,2.2,24.6,4.1,2.8,8.4,3.2,68.2,42.1,31.4,4.1,1.8,3.8,32000000000`,
  'asia-em': `ticker,sector,pe,pb,ev_ebitda,div_yield,fcf_yield,roe,roa,gross_margin,operating_margin,net_margin,rev_growth,debt_equity,interest_coverage
SAMSUNG,Technology,14.2,1.5,8.4,2.1,5.2,11.2,5.8,38.4,14.2,10.8,8.2,0.3,24.6
TSMC,Semiconductors,22.8,6.2,14.6,1.8,3.8,28.4,16.4,53.1,38.2,31.8,24.6,0.2,42.8
HYNIX,Semiconductors,18.4,1.8,9.2,1.2,4.1,14.8,6.2,32.6,18.4,12.6,35.4,0.6,18.2
LG,Consumer Electronics,11.4,0.9,7.8,3.2,6.1,8.2,3.4,24.1,7.8,5.2,3.8,0.8,8.4
MEDIATEK,Semiconductors,16.8,3.4,11.2,4.8,5.8,22.4,12.8,46.8,22.4,18.6,18.2,0.2,32.4,52000000000`,
};

// ── Sample benchmark CSVs — actual monthly returns 2023-01-31 ~ 2024-12-31 ─────────
// Source: Yahoo Finance / investing.com (monthly closing prices → returns)
const SAMPLE_BENCHMARKS: Record<string, string> = {
  'us-tech': `date,SP500
2023-01-31,0.0623
2023-02-28,-0.0265
2023-03-31,0.0376
2023-04-30,0.0145
2023-05-31,-0.0064
2023-06-30,0.0626
2023-07-31,0.0322
2023-08-31,-0.0177
2023-09-30,-0.0491
2023-10-31,-0.0221
2023-11-30,0.0887
2023-12-31,0.0453
2024-01-31,0.0159
2024-02-29,0.0521
2024-03-31,0.0322
2024-04-30,-0.0416
2024-05-31,0.0480
2024-06-30,0.0348
2024-07-31,0.0122
2024-08-31,0.0234
2024-09-30,0.0202
2024-10-31,-0.0099
2024-11-30,0.0566
2024-12-31,-0.0247`,
  'multi-asset': `date,MSCI_WORLD
2023-01-31,0.0701
2023-02-28,-0.0291
2023-03-31,0.0312
2023-04-30,0.0182
2023-05-31,-0.0102
2023-06-30,0.0598
2023-07-31,0.0341
2023-08-31,-0.0241
2023-09-30,-0.0432
2023-10-31,-0.0298
2023-11-30,0.0912
2023-12-31,0.0421
2024-01-31,0.0098
2024-02-29,0.0489
2024-03-31,0.0312
2024-04-30,-0.0381
2024-05-31,0.0421
2024-06-30,0.0289
2024-07-31,0.0134
2024-08-31,0.0198
2024-09-30,0.0187
2024-10-31,-0.0143
2024-11-30,0.0489
2024-12-31,-0.0198`,
  'asia-em': `date,KOSPI
2023-01-31,0.0842
2023-02-28,-0.0312
2023-03-31,0.0198
2023-04-30,0.0421
2023-05-31,-0.0187
2023-06-30,0.0312
2023-07-31,0.0198
2023-08-31,-0.0421
2023-09-30,-0.0312
2023-10-31,-0.0198
2023-11-30,0.0587
2023-12-31,0.0189
2024-01-31,-0.0421
2024-02-29,0.0312
2024-03-31,0.0198
2024-04-30,-0.0189
2024-05-31,0.0421
2024-06-30,0.0098
2024-07-31,0.0312
2024-08-31,-0.0198
2024-09-30,0.0421
2024-10-31,-0.0312
2024-11-30,-0.0587
2024-12-31,-0.0421`,
};

// Parse benchmark CSV: date + single return column
// Parse standalone fundamentals CSV
// Parse standalone fundamentals CSV (no ## FUNDAMENTALS header needed)
function parseStandaloneFundamentals(csv: string): Record<string, Record<string, number | string | null>> {
  const result: Record<string, Record<string, number | string | null>> = {};
  const COL_MAP: Record<string, string> = {
    // Value
    pe: 'pe', pb: 'pb',
    ev_ebitda: 'evEbitda', evebitda: 'evEbitda',
    div_yield: 'divYield', divyield: 'divYield', dividend_yield: 'divYield',
    fcf_yield: 'fcfYield', fcfyield: 'fcfYield',
    // Quality — profitability
    roe: 'roe',
    roa: 'roa',
    gross_margin: 'grossMargin', grossmargin: 'grossMargin',
    operating_margin: 'operatingMargin', operatingmargin: 'operatingMargin', op_margin: 'operatingMargin',
    net_margin: 'netMargin', netmargin: 'netMargin',
    rev_growth: 'revenueGrowth', revgrowth: 'revenueGrowth', revenue_growth: 'revenueGrowth',
    // Quality — leverage
    debt_equity: 'debtEquity', debtequity: 'debtEquity', de: 'debtEquity',
    interest_coverage: 'interestCoverage', interestcoverage: 'interestCoverage', int_coverage: 'interestCoverage',
    // Identity
    sector: 'sector', gics_sector: 'sector', industry: 'sector',
    market_cap: 'marketCap', marketcap: 'marketCap', mktcap: 'marketCap',
  };
  // Text fields — not parsed as float
  const TEXT_FIELDS = new Set(['sector']);

  const lines = csv.trim().split('\n').filter(l => l.trim());
  if (lines.length < 2) return result;
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const tickerIdx = headers.findIndex(h => h === 'ticker' || h === 'symbol');
  if (tickerIdx === -1) return result;
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(',').map(c => c.trim());
    const ticker = cells[tickerIdx]?.toUpperCase();
    if (!ticker) continue;
    result[ticker] = {};
    headers.forEach((h, j) => {
      if (j === tickerIdx) return;
      const field = COL_MAP[h];
      if (!field) return;
      const raw = cells[j];
      if (!raw || raw === '') {
        result[ticker][field] = null;
      } else if (TEXT_FIELDS.has(field)) {
        result[ticker][field] = raw;
      } else {
        result[ticker][field] = parseFloat(raw);
      }
    });
  }
  return result;
}

export default function ImportAssetsPage({ assets, onFileSelected, onNavigate, onAssetsChange, onBenchmarkChange, isUploading }: InvestmentPageProps) {
  const hasData = assets.length > 0;
  const [showFundamentals, setShowFundamentals] = useState(false);
  const [fundamentalInputs, setFundamentalInputs] = useState<Record<string, Record<string, string>>>({});
  const [sectorInputs, setSectorInputs] = useState<Record<string, string>>({});
  const [savedFundamentals, setSavedFundamentals] = useState(false);
  const [fundFileName, setFundFileName] = useState<string | null>(null);
  const [fundApplied, setFundApplied] = useState(false);

  // Handle benchmark CSV upload



  // Handle standalone fundamentals CSV upload
  const handleFundamentalsFile = (file: File) => {
    setFundFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const csv = e.target?.result as string;
      const parsed = parseStandaloneFundamentals(csv);
      if (!onAssetsChange || Object.keys(parsed).length === 0) return;
      onAssetsChange((prev: any[]) => {
        if (!prev.length) return prev;
        return prev.map((a: any) => {
          const fund = parsed[a.ticker];
          if (!fund) return a;
          return { ...a, ...fund };
        });
      });
      setFundApplied(true);
    };
    reader.readAsText(file);
  };

  // Load sample fundamentals
  const handleSampleFundamentals = (id: string) => {
    const csv = SAMPLE_FUNDAMENTALS[id];
    if (!csv) return;
    const parsed = parseStandaloneFundamentals(csv);
    if (!onAssetsChange) return;
    // Use functional updater so we always get the latest assets state,
    // not the stale closure value from when handleSample was called
    onAssetsChange((prev: any[]) => {
      if (!prev.length) return prev;
      return prev.map((a: any) => {
        const fund = parsed[a.ticker];
        if (!fund) return a;
        return { ...a, ...fund };
      });
    });
    setFundFileName(`sample-${id}-fundamentals.csv`);
    setFundApplied(true);
  };

  // Load sample benchmark
  const handleSampleBenchmark = (id: string) => {
    const csv = SAMPLE_BENCHMARKS[id];
    if (!csv || !onBenchmarkChange) return;
    const lines = csv.trim().split('\n');
    const header = lines[0].split(',');
    const nameCol = header[1]?.trim() ?? 'Benchmark';
    const periods: string[] = [];
    const returns: number[] = [];
    for (let i = 1; i < lines.length; i++) {
      const [date, val] = lines[i].split(',');
      if (!date || !val) continue;
      periods.push(date.trim());
      returns.push(parseFloat(val.trim()));
    }
    if (returns.length > 0) {
      onBenchmarkChange({ name: nameCol, returns, periods });
    }
  };

  const statsPerAsset = useMemo(() => assets.map(a => ({
    ...a, stats: calcStats(a.returns),
  })), [assets]);

  const handleSample = (s: typeof SAMPLES[0]) => {
    // Strip ## FUNDAMENTALS section — price CSV is now separate
    const priceOnly = s.csv.split('## FUNDAMENTALS')[0].trim();
    const file = new File([priceOnly], `sample-${s.id}-prices.csv`, { type: 'text/csv' });
    onFileSelected(file);
    // Load sample fundamentals + benchmark after assets are parsed
    // Use longer delay to ensure assets state is updated before fundamentals merge
    setTimeout(() => {
      handleSampleFundamentals(s.id);
      handleSampleBenchmark(s.id);
    }, 300);
  };

  // ── Fundamental input handlers ────────────────────────────────────────────
  const updateFundamental = (ticker: string, key: string, value: string) => {
    setFundamentalInputs(prev => ({
      ...prev,
      [ticker]: { ...prev[ticker], [key]: value },
    }));
    setSavedFundamentals(false);
  };

  const applyFundamentals = () => {
    if (!onAssetsChange) return;
    const currentInputs = fundamentalInputs; // capture current ref
    onAssetsChange((prev: any[]) => prev.map((a: any) => {
      const inputs = currentInputs[a.ticker] ?? {};
      const parse = (k: string) => {
        const v = inputs[k];
        if (v !== undefined && v !== '') return parseFloat(v);
        return a[k] ?? null;
      };
      return {
        ...a,
        pe: parse('pe'), pb: parse('pb'), evEbitda: parse('evEbitda'),
        divYield: parse('divYield'), fcfYield: parse('fcfYield'),
        roe: parse('roe'), roa: parse('roa'),
        grossMargin: parse('grossMargin'), operatingMargin: parse('operatingMargin'),
        netMargin: parse('netMargin'), revenueGrowth: parse('revenueGrowth'),
        debtEquity: parse('debtEquity'), interestCoverage: parse('interestCoverage'),
        // sector is handled separately via sectorInputs
        sector: sectorInputs[a.ticker] ?? a.sector ?? null,
      };
    }));
    setSavedFundamentals(true);
  };

  // Count how many assets have fundamentals already
  const assetsWithFundamentals = assets.filter(a => (a as any).roe !== undefined && (a as any).roe !== null).length;
  const assetsWithAnyFundamental = assets.filter(a => {
    const af = a as any;
    return FUNDAMENTAL_FIELDS.some(f => af[f.key] !== undefined && af[f.key] !== null);
  }).length;

  if (!hasData) {
    return (
      <div className="flex flex-col gap-8 max-w-6xl mx-auto">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Upload className="w-5 h-5 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">Import Assets</h1>
            <span className="ml-2 text-sm font-bold uppercase tracking-wider text-muted-foreground border border-border rounded-full px-2.5 py-0.5">Step 1 of 3</span>
          </div>
          <p className="text-sm text-muted-foreground">Upload your data files or load a sample portfolio instantly. Each section shows the required CSV format on the left and the upload zone on the right.</p>
        </div>

        {/* ① Price Data ─────────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Left: format guide */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center shrink-0">1</span>
              <p className="text-sm font-bold text-foreground">Price Data</p>
              <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-red-500/10 text-red-600 border border-red-200">REQUIRED</span>
            </div>
            <p className="text-sm text-muted-foreground">Wide format — one column per asset, one row per period. Date column on the left.</p>
            <div className="rounded-xl border border-border bg-card overflow-hidden">
              <div className="px-3 py-2 bg-muted/50 border-b border-border">
                <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider">prices.csv</p>
              </div>
              <table className="w-full text-sm">
                <thead><tr className="border-b border-border bg-muted/20">{['Date','AAPL','MSFT','GOOGL'].map(h => <th key={h} className="text-left py-2 px-3 font-bold text-muted-foreground">{h}</th>)}</tr></thead>
                <tbody>
                  {[['2024-01-31','185.2','374.5','140.9'],['2024-02-29','182.1','380.2','155.4'],['2024-03-31','171.2','420.5','160.6'],['2024-04-30','169.9','406.3','161.5']].map((row, i) => (
                    <tr key={i} className="border-b border-border/40 last:border-0">
                      {row.map((v, j) => <td key={j} className={cn('py-2 px-3 font-mono', j === 0 ? 'text-primary font-semibold' : 'text-foreground')}>{v}</td>)}
                    </tr>
                  ))}
                  <tr><td colSpan={4} className="py-1.5 px-3 text-sm text-muted-foreground">... more rows</td></tr>
                </tbody>
              </table>
            </div>
            <div className="flex flex-col gap-1 text-sm text-muted-foreground">
              <p>✓ Date formats: <code className="bg-muted px-1 rounded">2024-01-31</code> <code className="bg-muted px-1 rounded">2024-01-31-31</code> <code className="bg-muted px-1 rounded">Jan-2024</code></p>
              <p>✓ Also accepts long format: Ticker, Date, Price</p>
              <p>✓ Excel (.xlsx) and CSV both supported</p>
            </div>
          </div>
          {/* Right: upload zone */}
          <div className="flex flex-col gap-3">
            <div className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-primary/40 bg-primary/5 hover:border-primary/70 transition-all py-10 gap-4 cursor-pointer"
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onFileSelected(f); }}>
              <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center">
                <Database className="w-6 h-6 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-foreground mb-1">Drop price file here</p>
                <p className="text-sm text-muted-foreground">CSV or Excel · drag & drop or browse</p>
              </div>
              <label className="cursor-pointer inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors">
                <Upload className="w-4 h-4" />Browse Price File
                <input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onFileSelected(f); }} />
              </label>
            </div>
            {/* Samples */}
            <div>
              <p className="text-sm font-semibold text-muted-foreground mb-2">Or load a sample — loads all 3 files at once:</p>
              <div className="flex flex-col gap-2">
                {SAMPLES.map(s => (
                  <button key={s.id} onClick={() => handleSample(s)}
                    className="group flex items-center gap-3 p-3 rounded-xl border border-border bg-card hover:border-primary/40 hover:bg-primary/5 transition-all text-left">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: s.color + '20' }}>
                      <Database className="w-4 h-4" style={{ color: s.color }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold group-hover:text-primary transition-colors">{s.label}</p>
                      <p className="text-sm text-muted-foreground">{s.meta} · {s.desc}</p>
                    </div>
                    <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-green-500/10 text-green-700 border border-green-200 shrink-0">Price + Fund + Bmk ✓</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-border"/>

        {/* ② Fundamentals ───────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Left: format guide */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-muted border border-border text-muted-foreground text-xs font-bold flex items-center justify-center shrink-0">2</span>
              <p className="text-sm font-bold text-foreground">Fundamentals</p>
              <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">OPTIONAL</span>
            </div>
            <p className="text-sm text-muted-foreground">One row per ticker. Enables DCF, Multiples, Relative Valuation, Value Score, and Quality Score pages.</p>
            <div className="rounded-xl border border-border bg-card overflow-hidden">
              <div className="px-3 py-2 bg-muted/50 border-b border-border">
                <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider">fundamentals.csv</p>
              </div>
              <table className="w-full text-sm">
                <thead><tr className="border-b border-border bg-muted/20">{['ticker','pe','pb','roe','op_margin','net_margin'].map(h => <th key={h} className="text-left py-2 px-3 font-bold text-muted-foreground">{h}</th>)}</tr></thead>
                <tbody>
                  {[['AAPL','28.5','45.2','160.1','29.2','23.9'],['MSFT','35.2','12.8','39.2','44.6','34.1'],['GOOGL','24.1','6.8','27.3','26.5','23.7']].map((row, i) => (
                    <tr key={i} className="border-b border-border/40 last:border-0">
                      {row.map((v, j) => <td key={j} className={cn('py-2 px-3 font-mono', j === 0 ? 'text-primary font-semibold' : 'text-foreground')}>{v}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="grid grid-cols-2 gap-1">
              {[
                ['ticker / symbol', 'Ticker identifier'],
                ['sector',          'Sector (text) — for relative valuation'],
                ['pe',              'P/E ratio'],
                ['pb',              'P/B ratio'],
                ['ev_ebitda',       'EV/EBITDA'],
                ['div_yield',       'Dividend yield %'],
                ['fcf_yield',       'FCF yield %'],
                ['roe',             'Return on equity %'],
                ['roa',             'Return on assets %'],
                ['gross_margin',    'Gross margin %'],
                ['operating_margin','Operating margin %'],
                ['net_margin',      'Net margin %'],
                ['rev_growth',      'Revenue growth %'],
                ['debt_equity',     'D/E ratio'],
                ['interest_coverage','Interest coverage x'],
                ['market_cap',      'Market cap (USD)'],
              ].map(([col, desc]) => (
                <div key={col} className="flex items-center gap-1.5">
                  <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono text-primary">{col}</code>
                  <span className="text-sm text-muted-foreground">{desc}</span>
                </div>
              ))}
            </div>
          </div>
          {/* Right: upload zone */}
          <div className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-muted/10 hover:border-primary/30 transition-all py-10 gap-4"
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) handleFundamentalsFile(f); }}>
            <div className="w-12 h-12 rounded-2xl bg-muted flex items-center justify-center">
              <FileSpreadsheet className="w-6 h-6 text-muted-foreground" />
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-foreground mb-1">Drop fundamentals file here</p>
              <p className="text-sm text-muted-foreground">CSV only · upload anytime, even after price data</p>
            </div>
            <label className="cursor-pointer inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-border text-sm font-semibold text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
              <Upload className="w-4 h-4" />Browse Fundamentals File
              <input type="file" accept=".csv" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFundamentalsFile(f); }} />
            </label>
          </div>
        </div>

      </div>
    );
  }
  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Upload className="w-5 h-5 text-primary" />
          <h1 className="text-2xl font-bold text-foreground">Import Assets</h1>
          <span className="ml-2 text-sm font-bold uppercase tracking-wider text-muted-foreground border border-border rounded-full px-2.5 py-0.5">Step 1 of 3</span>
        </div>
        <p className="text-sm text-muted-foreground">Review loaded assets before proceeding to portfolio configuration.</p>
      </div>

      {/* Success banners — price + fundamentals + benchmark status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Price data status */}
        <div className="flex items-center gap-3 p-4 rounded-xl border border-green-500/30 bg-green-500/5">
          <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-foreground">{assets.length} asset{assets.length > 1 ? 's' : ''} loaded</p>
            <p className="text-sm text-muted-foreground truncate">{assets.map(a => a.ticker).join(' · ')}</p>
          </div>
          <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-sm font-medium text-muted-foreground hover:bg-muted transition-colors shrink-0">
            <RefreshCw className="w-3.5 h-3.5" />Replace
            <input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) onFileSelected(f); }} />
          </label>
        </div>
        {/* Fundamentals status */}
        <div className={cn('flex items-center gap-3 p-4 rounded-xl border transition-all',
          fundApplied ? 'border-green-500/30 bg-green-500/5' : 'border-dashed border-border bg-muted/10 hover:border-primary/30')}>
          {fundApplied
            ? <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
            : <FileSpreadsheet className="w-5 h-5 text-muted-foreground shrink-0" />}
          <div className="flex-1 min-w-0">
            {fundApplied
              ? <><p className="text-sm font-bold text-foreground">{assetsWithAnyFundamental}/{assets.length} with fundamentals</p>
                  <p className="text-sm text-muted-foreground truncate">{fundFileName}</p></>
              : <><p className="text-sm font-semibold text-muted-foreground">No fundamentals loaded</p>
                  <p className="text-sm text-muted-foreground">Optional — enables DCF, Multiples, Relative Valuation</p></>}
          </div>
          <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-sm font-medium text-muted-foreground hover:bg-muted transition-colors shrink-0">
            <Upload className="w-3.5 h-3.5" />{fundApplied ? 'Replace' : 'Upload'}
            <input type="file" accept=".csv" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFundamentalsFile(f); }} />
          </label>
        </div>
        {/* Benchmark — set in Benchmark step */}
        <div className="flex items-center gap-3 p-4 rounded-xl border border-dashed border-border bg-muted/10">
          <Target className="w-5 h-5 text-muted-foreground shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-muted-foreground">Benchmark set in Step 3</p>
            <p className="text-sm text-muted-foreground">Choose a live index or upload CSV on the Benchmark page.</p>
          </div>
          <button onClick={() => onNavigate?.('benchmark')}
            className="text-sm font-bold text-primary hover:underline shrink-0">
            Go →
          </button>
        </div>
      </div>

      {/* Missing fundamentals notice */}
      {assetsWithAnyFundamental < assets.length && (
        <div className="flex items-start gap-3 px-4 py-3.5 rounded-xl border border-amber-300 bg-amber-500/5">
          <Info className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="flex-1 text-sm text-amber-700">
            <span className="font-semibold">{assets.length - assetsWithAnyFundamental} asset{assets.length - assetsWithAnyFundamental > 1 ? 's' : ''} without fundamental data</span>
            {' '}— Value Score, Quality Score, and Factor Exposure will show partial results. Upload a Fundamentals CSV using the upload button above, or add manually below.
          </div>
          <button onClick={() => setShowFundamentals(true)}
            className="text-sm font-bold text-amber-700 underline shrink-0">Add now</button>
        </div>
      )}

      {/* Asset summary table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg font-semibold">Asset Summary</CardTitle>
          <CardDescription className="text-sm">Key statistics from historical price data</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/50">
                  {['Ticker', 'Periods', 'Latest Price', 'Total Return', 'Avg Return', 'Volatility', 'Sharpe', 'Max Drawdown', 'Sector', 'Fundamentals'].map(h => (
                    <th key={h} className={`px-4 py-3 font-semibold text-muted-foreground uppercase tracking-wide text-sm ${h === 'Ticker' || h === 'Fundamentals' || h === 'Sector' ? 'text-left' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {statsPerAsset.map(a => {
                  const af = a as any;
                  const fundFields = FUNDAMENTAL_FIELDS.filter(f => af[f.key] !== undefined && af[f.key] !== null);
                  return (
                    <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/20 transition-colors">
                      <td className="px-4 py-3 font-bold text-primary text-base">{a.ticker}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{a.prices.length}</td>
                      <td className="px-4 py-3 text-right font-mono text-foreground">{fmt(a.prices[a.prices.length - 1] ?? 0)}</td>
                      <td className="px-4 py-3 text-right">
                        <span className={cn('font-mono font-bold text-xs px-1.5 py-0.5 rounded', a.stats.totalReturn >= 0 ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-500')}>
                          {pctFmt(a.stats.totalReturn)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono" style={{ color: a.stats.mean >= 0 ? '#10B981' : '#EF4444' }}>{pctFmt(a.stats.mean)}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{(a.stats.std * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right font-mono" style={{ color: a.stats.sharpe >= 1 ? '#10B981' : a.stats.sharpe >= 0 ? '#F59E0B' : '#EF4444' }}>{a.stats.sharpe.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right font-mono text-red-500">-{(a.stats.maxDrawdown * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-muted-foreground text-sm">
                        {(a as any).sector
                          ? <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">{(a as any).sector}</span>
                          : <span className="text-muted-foreground">—</span>}
                      </td>
                      <td className="px-4 py-3">
                        {fundFields.length > 0
                          ? <span className="text-sm font-bold px-2 py-0.5 rounded bg-green-500/10 text-green-700 border border-green-200">{fundFields.length}/{FUNDAMENTAL_FIELDS.length} fields</span>
                          : <span className="text-sm text-muted-foreground">—</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Fundamentals manual input panel */}
      <Card className={showFundamentals ? 'border-primary/30' : ''}>
        <CardHeader className="pb-3 cursor-pointer select-none" onClick={() => setShowFundamentals(s => !s)}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Plus className="w-5 h-5 text-primary" />
              <CardTitle className="text-lg font-semibold">Add Fundamental Data</CardTitle>
              {assetsWithAnyFundamental > 0 && (
                <span className="text-sm font-bold px-2.5 py-0.5 rounded-full bg-green-500/10 text-green-700 border border-green-200">{assetsWithAnyFundamental}/{assets.length} assets</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {savedFundamentals && <span className="text-sm text-green-600 font-semibold">✓ Applied</span>}
              {showFundamentals ? <ChevronUp className="w-5 h-5 text-muted-foreground" /> : <ChevronDown className="w-5 h-5 text-muted-foreground" />}
            </div>
          </div>
          <CardDescription className="text-sm">Manually enter P/E, P/B, ROE, margins etc. — enables Value Score, Quality Score, and Factor Exposure</CardDescription>
        </CardHeader>

        {showFundamentals && (
          <CardContent className="space-y-4 pt-0">
            <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/40 border border-border">
              <Info className="w-3.5 h-3.5 text-muted-foreground shrink-0 mt-0.5" />
              <p className="text-sm text-muted-foreground">
                Leave blank to skip. Percentages: enter as a number (e.g. <code className="bg-muted px-1 rounded">18</code> for 18%). Ratios: enter as a decimal (e.g. <code className="bg-muted px-1 rounded">0.5</code> for 0.5x D/E).
                Or upload a separate Fundamentals CSV (ticker, pe, pb, roe...) using the upload button above — no special headers needed.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/50">
                    <th className="px-3 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wide w-24">Ticker</th>
                    <th className="px-2 py-3 text-center font-semibold text-muted-foreground uppercase tracking-wide min-w-[120px]">
                      <div>Sector</div>
                      <div className="text-xs font-normal normal-case text-muted-foreground/70">text</div>
                    </th>
                    {FUNDAMENTAL_FIELDS.map(f => (
                      <th key={f.key} className="px-2 py-3 text-center font-semibold text-muted-foreground uppercase tracking-wide min-w-[90px]">
                        <div>{f.label}</div>
                        <div className="text-xs font-normal normal-case text-muted-foreground/70">{f.unit}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {assets.map(a => {
                    const af = a as any;
                    return (
                      <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/10 transition-colors">
                        <td className="px-3 py-3 font-bold text-primary text-base">{a.ticker}</td>
                        {/* Sector — text input */}
                        <td className="px-1.5 py-2.5">
                          <input
                            type="text"
                            placeholder="e.g. Technology"
                            value={sectorInputs[a.ticker] ?? (a as any).sector ?? ''}
                            onChange={e => setSectorInputs(prev => ({ ...prev, [a.ticker]: e.target.value }))}
                            title="Sector — used for sector-relative valuation and allocation charts"
                            className={cn(
                              'w-full rounded-lg border px-2 py-1.5 text-sm font-mono text-left focus:outline-none focus:ring-1 focus:ring-primary bg-background',
                              (sectorInputs[a.ticker] ?? (a as any).sector) ? 'border-primary/40 text-foreground' : 'border-border text-muted-foreground'
                            )}
                          />
                        </td>
                        {FUNDAMENTAL_FIELDS.map(f => {
                          const existingVal = af[f.key];
                          const inputVal = fundamentalInputs[a.ticker]?.[f.key];
                          const displayVal = inputVal ?? (existingVal !== null && existingVal !== undefined ? String(existingVal) : '');
                          return (
                            <td key={f.key} className="px-1.5 py-2">
                              <input
                                type="number"
                                step="any"
                                placeholder={f.placeholder}
                                value={displayVal}
                                onChange={e => updateFundamental(a.ticker, f.key, e.target.value)}
                                title={`${f.label} — ${f.hint}`}
                                className={cn(
                                  'w-full rounded-lg border px-2 py-1.5 text-sm font-mono text-right focus:outline-none focus:ring-1 focus:ring-primary bg-background',
                                  displayVal ? 'border-primary/40 text-foreground' : 'border-border text-muted-foreground'
                                )}
                              />
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between pt-1">
              <p className="text-sm text-muted-foreground">Changes apply to all analysis pages immediately</p>
              <button onClick={applyFundamentals}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors">
                <CheckCircle2 className="w-3.5 h-3.5" />Apply Fundamentals
              </button>
            </div>
          </CardContent>
        )}
      </Card>

      {assets.some(a => a.prices.length < 12) && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3">
          <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-700">Some assets have fewer than 12 periods — statistical measures may be unreliable. Upload longer price history for accurate risk analysis.</p>
        </div>
      )}

      <div className="flex justify-end">
        <button onClick={() => onNavigate?.('portfolio-composition')}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors">
          Configure Portfolio <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}