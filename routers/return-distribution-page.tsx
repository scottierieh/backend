'use client';

import React, { useState, useEffect, useRef } from 'react';
import { BarChart3, Activity, Info, AlertCircle, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, LineChart, Line, ScatterChart, Scatter, ReferenceLine, ComposedChart, Area,
} from 'recharts';
import { cn } from '@/lib/utils';
import { type InvestmentPageProps } from '@/components/investment-intelligence-app';
import { useInvestmentAPI } from '@/hooks/useInvestmentAPI';

// ── Design tokens ─────────────────────────────────────────────────────────────
const COLORS  = ['#6C3AED','#10B981','#F59E0B','#EF4444','#3B82F6','#EC4899','#8B5CF6','#14B8A6'];
const C_GREEN = '#10B981';
const C_RED   = '#EF4444';
const C_AMBER = '#F59E0B';
const C_MUTED = '#94A3B8';
const C_PURP  = '#6C3AED';
const GRID    = '#F1F5F9';

// ── Types ─────────────────────────────────────────────────────────────────────
interface TailProbs  { lt_1pct: number; lt_2pct: number; lt_5pct: number; }
interface ExtremeFreq { threshold: number; count: number; pct: number; }
interface RegimeSplit {
  positive: { count: number; pct: number; mean: number | null; vol: number | null };
  negative: { count: number; pct: number; mean: number | null; vol: number | null };
}
interface VolRegime {
  highVol: { mean: number | null; skew: number | null; kurt: number | null; count: number };
  lowVol:  { mean: number | null; skew: number | null; kurt: number | null; count: number };
}
interface Boxplot {
  min: number; q1: number; median: number; q3: number; max: number;
  iqr: number; fenceLo: number; fenceHi: number; outliers: number[];
}
interface AssetStats {
  ticker:        string;
  mean:          number;
  std:           number;
  skewness:      number;
  kurtosis:      number;
  jb:            number;
  jbP:           number;
  normal:        boolean;
  avgGain:       number;
  avgLoss:       number;
  gainLossRatio: number | null;
  downsideDev:   number;
  tailProbs:     TailProbs;
  extremeFreq:   ExtremeFreq[];
  tailRatio:     number | null;
  regimeSplit:   RegimeSplit;
  volRegime:     VolRegime;
  lossHist:      { range: string; count: number }[];
  assetHist:     { range: string; midpoint: number; count: number }[];
  boxplot:       Boxplot;
}
interface DistResult {
  histogram:   { range: string; midpoint: number; count: number; isTail: boolean }[];
  normalCurve: { x: number; pdf: number }[];
  qqData:      { theoretical: number; sample: number }[];
  qqLine:      { theoretical: number; ref: number }[];
  portfolio:   AssetStats;
  assets:      AssetStats[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const pct  = (v: number | null | undefined, d = 2) => v != null ? `${(v * 100).toFixed(d)}%` : '—';
const fmt  = (v: number | null | undefined, d = 3) => v != null ? v.toFixed(d) : '—';
const signed = (v: number | null | undefined, d = 2) => v != null ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%` : '—';

function InsightRow({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-0.5 min-h-[44px] rounded-full shrink-0 mt-1 bg-primary/30" />
      <div>
        <p className="text-sm font-semibold mb-0.5 text-primary">{title}</p>
        <p className="text-sm text-muted-foreground leading-relaxed">{children}</p>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function KpiStrip({ port, loading }: { port: AssetStats | undefined; loading: boolean }) {
  const skew = port?.skewness ?? 0;
  const kurt = port?.kurtosis ?? 0;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {[
        {
          label: 'Skewness',
          value: fmt(port?.skewness),
          color: Math.abs(skew) > 0.5 ? (skew < 0 ? C_RED : C_AMBER) : C_GREEN,
          note: skew < -0.5 ? 'Left tail — loss asymmetry' : skew > 0.5 ? 'Right tail' : 'Approximately symmetric',
        },
        {
          label: 'Excess Kurtosis',
          value: fmt(port?.kurtosis),
          color: kurt > 1 ? C_RED : kurt > 0 ? C_AMBER : C_GREEN,
          note: kurt > 1 ? 'Fat tails — extreme events likely' : kurt > 0 ? 'Slightly fat tails' : 'Thin tails',
        },
        {
          label: 'JB p-value',
          value: fmt(port?.jbP),
          color: (port?.jbP ?? 1) < 0.05 ? C_RED : C_GREEN,
          note: (port?.jbP ?? 1) < 0.05 ? 'Non-normal (p < 0.05)' : 'Cannot reject normality',
        },
        {
          label: 'Tail Ratio',
          value: port?.tailRatio != null ? port.tailRatio.toFixed(2) : '—',
          color: (port?.tailRatio ?? 1) >= 1 ? C_GREEN : C_RED,
          note: 'Right tail / Left tail',
        },
      ].map(s => (
        <Card key={s.label}><CardContent className="p-4">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">{s.label}</p>
          <p className="text-2xl font-bold" style={{ color: s.color }}>{loading ? '…' : s.value}</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">{s.note}</p>
        </CardContent></Card>
      ))}
    </div>
  );
}


function HistogramWithNormal({
  hist, normalCurve, title,
}: {
  hist: DistResult['histogram'];
  normalCurve: DistResult['normalCurve'];
  title: string;
}) {
  // Merge histogram + normal curve by midpoint for ComposedChart
  const data = hist.map(h => ({
    range:  h.range,
    mid:    h.midpoint,
    count:  h.count,
    isTail: h.isTail,
    // Find closest normal curve point
    normal: (() => {
      if (!normalCurve.length) return null;
      let closest = normalCurve[0];
      for (const nc of normalCurve) {
        if (Math.abs(nc.x - h.midpoint) < Math.abs(closest.x - h.midpoint)) closest = nc;
      }
      return closest.pdf;
    })(),
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">{title}</CardTitle>
        <CardDescription className="text-xs">Red = tail bins (below 5th pct) · Dashed = normal overlay</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey="range" tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false}
              interval={Math.floor(data.length / 8)} />
            <YAxis tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false} />
            <Tooltip formatter={(v: any, name: string) => [
              typeof v === 'number' ? (name === 'normal' ? v.toFixed(1) : v) : v,
              name === 'normal' ? 'Normal' : 'Count',
            ]} />
            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.isTail ? C_RED : C_PURP} opacity={d.isTail ? 0.85 : 0.55} />
              ))}
            </Bar>
            <Line type="monotone" dataKey="normal" stroke={C_AMBER} strokeWidth={2}
              dot={false} strokeDasharray="5 3" name="normal" />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}


function QQPlot({ qqData, qqLine }: { qqData: DistResult['qqData']; qqLine: DistResult['qqLine'] }) {
  if (!qqData.length) return null;

  // Build reference line from qqLine endpoints
  const refMin = qqLine[0]?.theoretical ?? qqData[0]?.theoretical ?? 0;
  const refMax = qqLine[1]?.theoretical ?? qqData[qqData.length - 1]?.theoretical ?? 1;

  const refData = [
    { theoretical: refMin, ref: refMin },
    { theoretical: refMax, ref: refMax },
  ];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Q-Q Plot</CardTitle>
        <CardDescription className="text-xs">Points on diagonal = normal · Heavy tails → S-curve deviation</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
            <XAxis dataKey="theoretical" type="number" name="Theoretical"
              label={{ value: 'Theoretical Quantile', position: 'insideBottom', offset: -12, fontSize: 10, fill: C_MUTED }}
              tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false}
              tickFormatter={v => pct(v)} />
            <YAxis dataKey="sample" type="number" name="Sample"
              label={{ value: 'Sample Quantile', angle: -90, position: 'insideLeft', fontSize: 10, fill: C_MUTED }}
              tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false}
              tickFormatter={v => pct(v)} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-background border border-border rounded-lg px-3 py-2 text-xs shadow-md">
                    <p className="text-muted-foreground">Theoretical: {pct(d.theoretical)}</p>
                    <p className="text-muted-foreground">Sample: {pct(d.sample)}</p>
                  </div>
                );
              }}
            />
            {/* Reference line (normal) */}
            <Scatter data={refData} dataKey="ref" line={{ stroke: C_AMBER, strokeWidth: 1.5, strokeDasharray: '4 3' }}
              shape={() => null as any} legendType="none" />
            {/* Sample points */}
            <Scatter data={qqData} dataKey="sample" fill={C_PURP} opacity={0.65} r={3} />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}


function LossOnlyChart({ lossHist, title }: { lossHist: AssetStats['lossHist']; title: string }) {
  if (!lossHist.length) return null;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Loss Distribution — {title}</CardTitle>
        <CardDescription className="text-xs">Negative returns only · left tail focus</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={lossHist} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey="range" tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false}
              interval={Math.floor(lossHist.length / 6)} />
            <YAxis tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false} />
            <Tooltip formatter={(v: any) => [v, 'Periods']} />
            <Bar dataKey="count" fill={C_RED} opacity={0.75} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}


function BoxPlotChart({ assets }: { assets: AssetStats[] }) {
  if (!assets.length) return null;

  // Recharts doesn't have native box plots — approximate with stacked bars
  // Layers: min→fenceLo (transparent), fenceLo→q1, q1→median, median→q3, q3→fenceHi
  const data = assets.map(a => {
    const bp = a.boxplot;
    return {
      ticker:    a.ticker,
      fenceLo:   bp.fenceLo,
      // bar segments (stacked from fenceLo)
      seg1: bp.q1    - bp.fenceLo,   // fenceLo → Q1
      seg2: bp.median - bp.q1,        // Q1 → median
      seg3: bp.q3    - bp.median,     // median → Q3
      seg4: bp.fenceHi - bp.q3,       // Q3 → fenceHi
      median: bp.median,
    };
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Box Plot — Return Distribution</CardTitle>
        <CardDescription className="text-xs">Whiskers = 1.5× IQR · Centre line = median</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={Math.max(180, assets.length * 52)}>
          <BarChart data={data} layout="vertical" margin={{ left: 50, right: 50, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false}
              tickFormatter={v => pct(v)} />
            <YAxis type="category" dataKey="ticker" tick={{ fontSize: 11, fill: C_MUTED }} tickLine={false} width={45} />
            <ReferenceLine x={0} stroke="#E2E8F0" />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                const bp = assets.find(a => a.ticker === d.ticker)?.boxplot;
                if (!bp) return null;
                return (
                  <div className="bg-background border border-border rounded-lg px-3 py-2 text-xs shadow-md">
                    <p className="font-bold mb-1">{d.ticker}</p>
                    <p className="text-muted-foreground">Min: {pct(bp.min)}</p>
                    <p className="text-muted-foreground">Q1: {pct(bp.q1)}</p>
                    <p className="font-semibold">Median: {pct(bp.median)}</p>
                    <p className="text-muted-foreground">Q3: {pct(bp.q3)}</p>
                    <p className="text-muted-foreground">Max: {pct(bp.max)}</p>
                    <p className="text-muted-foreground">IQR: {pct(bp.iqr)}</p>
                  </div>
                );
              }}
            />
            {/* Transparent base to fenceLo */}
            <Bar dataKey="fenceLo" stackId="box" fill="transparent" />
            {/* fenceLo → Q1 */}
            <Bar dataKey="seg1" stackId="box" fill={C_MUTED} opacity={0.3} />
            {/* Q1 → median (darker) */}
            <Bar dataKey="seg2" stackId="box" fill={C_PURP} opacity={0.6} />
            {/* median → Q3 */}
            <Bar dataKey="seg3" stackId="box" fill={C_PURP} opacity={0.4} />
            {/* Q3 → fenceHi */}
            <Bar dataKey="seg4" stackId="box" fill={C_MUTED} opacity={0.3} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}


function MultiAssetOverlay({ assets }: { assets: AssetStats[] }) {
  if (assets.length < 2) return null;

  // Normalise each asset's histogram to density so scales match
  const allMids = Array.from(new Set(
    assets.flatMap(a => a.assetHist.map(h => h.midpoint))
  )).sort((a, b) => a - b);

  const data = allMids.map(mid => {
    const row: Record<string, any> = { mid: pct(mid) };
    for (const a of assets) {
      const bin = a.assetHist.find(h => h.midpoint === mid);
      row[a.ticker] = bin?.count ?? 0;
    }
    return row;
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Distribution Overlay — All Assets</CardTitle>
        <CardDescription className="text-xs">Histogram counts per return bin · compare shape across assets</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey="mid" tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false}
              interval={Math.floor(data.length / 8)} />
            <YAxis tick={{ fontSize: 9, fill: C_MUTED }} tickLine={false} axisLine={false} />
            <Tooltip />
            <ReferenceLine x="0.00%" stroke="#E2E8F0" />
            {assets.map((a, i) => (
              <Bar key={a.ticker} dataKey={a.ticker} fill={COLORS[i % COLORS.length]} opacity={0.6} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap gap-3 mt-2 justify-center">
          {assets.map((a, i) => (
            <div key={a.ticker} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              <span className="text-xs text-muted-foreground">{a.ticker}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}


function TailRiskTable({ assets }: { assets: AssetStats[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Left Tail Probability</CardTitle>
        <CardDescription className="text-xs">Empirical probability of return falling below threshold</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-xs">
          <thead><tr className="border-b border-border bg-muted/50">
            {['Asset', 'P(r < -1%)', 'P(r < -2%)', 'P(r < -5%)', 'Tail Ratio', 'Downside Dev'].map(h => (
              <th key={h} className={`px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide ${h === 'Asset' ? 'text-left' : 'text-right'}`}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {assets.map((a, i) => (
              <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/20">
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="font-bold text-foreground">{a.ticker}</span>
                  </div>
                </td>
                {[a.tailProbs.lt_1pct, a.tailProbs.lt_2pct, a.tailProbs.lt_5pct].map((v, j) => (
                  <td key={j} className="px-3 py-2.5 text-right font-mono"
                    style={{ color: v > 0.15 ? C_RED : v > 0.08 ? C_AMBER : C_GREEN }}>
                    {pct(v)}
                  </td>
                ))}
                <td className="px-3 py-2.5 text-right font-mono"
                  style={{ color: (a.tailRatio ?? 1) >= 1 ? C_GREEN : C_RED }}>
                  {a.tailRatio != null ? a.tailRatio.toFixed(2) : '—'}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">
                  {pct(a.downsideDev)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}


function ExtremeLossTable({ assets }: { assets: AssetStats[] }) {
  // Collect all unique thresholds
  const thresholds = assets[0]?.extremeFreq.map(e => e.threshold) ?? [];
  if (!thresholds.length) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Extreme Loss Frequency</CardTitle>
        <CardDescription className="text-xs">Number and % of periods with extreme negative returns</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-xs">
          <thead><tr className="border-b border-border bg-muted/50">
            <th className="px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide text-left">Asset</th>
            {thresholds.map(t => (
              <th key={t} className="px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide text-right">
                r &lt; {pct(t, 0)}
              </th>
            ))}
            <th className="px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide text-right">Avg Gain</th>
            <th className="px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide text-right">Avg Loss</th>
            <th className="px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide text-right">G/L Ratio</th>
          </tr></thead>
          <tbody>
            {assets.map((a, i) => (
              <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/20">
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="font-bold text-foreground">{a.ticker}</span>
                  </div>
                </td>
                {a.extremeFreq.map((e, j) => (
                  <td key={j} className="px-3 py-2.5 text-right font-mono"
                    style={{ color: e.pct > 0.1 ? C_RED : e.pct > 0.05 ? C_AMBER : C_MUTED }}>
                    {e.count} <span className="text-muted-foreground">({pct(e.pct, 1)})</span>
                  </td>
                ))}
                <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_GREEN }}>{signed(a.avgGain)}</td>
                <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_RED }}>{signed(a.avgLoss)}</td>
                <td className="px-3 py-2.5 text-right font-mono"
                  style={{ color: (a.gainLossRatio ?? 0) >= 1 ? C_GREEN : C_RED }}>
                  {a.gainLossRatio != null ? a.gainLossRatio.toFixed(2) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}


function RegimeTable({ assets }: { assets: AssetStats[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Positive vs Negative Period Split</CardTitle>
        <CardDescription className="text-xs">Return distribution differs across market states</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-xs">
          <thead><tr className="border-b border-border bg-muted/50">
            {['Asset', 'Up %', 'Up Mean', 'Up Vol', 'Down %', 'Down Mean', 'Down Vol'].map(h => (
              <th key={h} className={`px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide ${h === 'Asset' ? 'text-left' : 'text-right'}`}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {assets.map((a, i) => {
              const { positive: pos, negative: neg } = a.regimeSplit;
              return (
                <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/20">
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                      <span className="font-bold text-foreground">{a.ticker}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_GREEN }}>{pct(pos.pct, 0)}</td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_GREEN }}>{signed(pos.mean)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">{pct(pos.vol)}</td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_RED }}>{pct(neg.pct, 0)}</td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_RED }}>{signed(neg.mean)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">{pct(neg.vol)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}


function VolRegimeTable({ assets }: { assets: AssetStats[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">High vs Low Volatility Regime</CardTitle>
        <CardDescription className="text-xs">Distribution shape changes under different vol environments</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-xs">
          <thead><tr className="border-b border-border bg-muted/50">
            {['Asset', 'Regime', 'Count', 'Mean', 'Skewness', 'Kurtosis'].map(h => (
              <th key={h} className={`px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide ${['Asset', 'Regime'].includes(h) ? 'text-left' : 'text-right'}`}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {assets.flatMap((a, i) => [
              { label: 'High Vol', data: a.volRegime.highVol, color: C_RED },
              { label: 'Low Vol',  data: a.volRegime.lowVol,  color: C_GREEN },
            ].map(({ label, data, color }) => (
              <tr key={`${a.ticker}-${label}`} className="border-b border-border/50 last:border-0 hover:bg-muted/20">
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="font-bold text-foreground">{a.ticker}</span>
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                    style={{ color, backgroundColor: color + '20' }}>{label}</span>
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">{data.count}</td>
                <td className="px-3 py-2.5 text-right font-mono"
                  style={{ color: data.mean != null && data.mean >= 0 ? C_GREEN : C_RED }}>
                  {signed(data.mean)}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">{fmt(data.skew)}</td>
                <td className="px-3 py-2.5 text-right font-mono"
                  style={{ color: (data.kurt ?? 0) > 1 ? C_RED : C_MUTED }}>
                  {fmt(data.kurt)}
                </td>
              </tr>
            )))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}


function DistributionSummaryTable({ assets }: { assets: AssetStats[] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold">Distribution Summary</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="border-b border-border bg-muted/50">
              {['Asset', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max', 'Normal?'].map(h => (
                <th key={h} className={`px-3 py-2.5 font-semibold text-muted-foreground uppercase tracking-wide whitespace-nowrap ${h === 'Asset' ? 'text-left' : 'text-right'}`}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {assets.map((a, i) => (
                <tr key={a.ticker} className="border-b border-border/50 last:border-0 hover:bg-muted/20">
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                      <span className="font-bold text-foreground">{a.ticker}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono"
                    style={{ color: a.mean >= 0 ? C_GREEN : C_RED }}>{signed(a.mean)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">{pct(a.std)}</td>
                  <td className="px-3 py-2.5 text-right font-mono"
                    style={{ color: Math.abs(a.skewness) > 0.5 ? C_AMBER : C_MUTED }}>
                    {a.skewness.toFixed(3)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono"
                    style={{ color: a.kurtosis > 1 ? C_RED : C_MUTED }}>
                    {a.kurtosis.toFixed(3)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_RED }}>
                    {pct(a.boxplot.min)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono" style={{ color: C_GREEN }}>
                    {pct(a.boxplot.max)}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <span className={cn('text-[10px] font-bold px-1.5 py-0.5 rounded-full',
                      a.normal ? 'bg-green-500/10 text-green-600' : 'bg-red-500/10 text-red-500')}>
                      {a.normal ? 'Yes' : 'No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}


// ── Main page ─────────────────────────────────────────────────────────────────

export default function ReturnDistributionPage({ assets, onNavigate }: InvestmentPageProps) {
  const api = useInvestmentAPI();

  const [logScale, setLogScale] = useState(false);
  const [nBins,    setNBins]    = useState(30);

  const [result,  setResult]  = useState<DistResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!assets.length) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError('');
    api.statistics.distribution({
      assets, nBins, logScale,
      extremeThresholds: [-0.03, -0.05],
    })
      .then(r => setResult(r as DistResult))
      .catch(e => { if (e.name !== 'AbortError') setError(e.message ?? 'API error'); })
      .finally(() => setLoading(false));
  }, [assets, nBins, logScale]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!assets.length) return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-primary" />
        <h1 className="text-xl font-bold">Return Distribution</h1>
      </div>
      <div className="flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3">
        <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
        <p className="text-sm text-amber-700">
          No assets loaded.{' '}
          <button onClick={() => onNavigate?.('import-assets')} className="underline font-semibold">
            Import assets first.
          </button>
        </p>
      </div>
    </div>
  );

  const port       = result?.portfolio;
  const assetStats = result?.assets ?? [];

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto">

      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h1 className="text-xl font-bold">Return Distribution</h1>
          {loading && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
        </div>
        <p className="text-sm text-muted-foreground">
          Distribution shape, tail risk, regime analysis, Q-Q plot, and multi-asset comparison.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
          <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-border overflow-hidden text-xs font-medium">
          {(['simple', 'log'] as const).map(t => (
            <button key={t} onClick={() => setLogScale(t === 'log')}
              className={cn('px-3 py-1.5 capitalize transition-colors',
                logScale === (t === 'log') ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted')}>
              {t}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-muted-foreground font-medium">Bins:</span>
          {[20, 30, 40].map(b => (
            <button key={b} onClick={() => setNBins(b)}
              className={cn('px-2.5 py-1.5 rounded border text-xs font-bold transition-colors',
                nBins === b ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/40')}>
              {b}
            </button>
          ))}
        </div>
      </div>

      {/* KPIs */}
      <KpiStrip port={port} loading={loading} />

      {result && (
        <>
          {/* Row 1: Histogram + Normal overlay | Q-Q Plot */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <HistogramWithNormal
              hist={result.histogram}
              normalCurve={result.normalCurve}
              title="Portfolio Return Histogram"
            />
            <QQPlot qqData={result.qqData} qqLine={result.qqLine} />
          </div>

          {/* Row 2: Loss-only chart | Multi-asset overlay */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {port && <LossOnlyChart lossHist={port.lossHist} title="Portfolio" />}
            <MultiAssetOverlay assets={assetStats} />
          </div>

          {/* Box Plot */}
          {assetStats.length > 0 && <BoxPlotChart assets={assetStats} />}

          {/* Distribution Summary Table */}
          {assetStats.length > 0 && <DistributionSummaryTable assets={assetStats} />}

          {/* Tail Risk Table */}
          {assetStats.length > 0 && <TailRiskTable assets={assetStats} />}

          {/* Extreme Loss + Gain-Loss Table */}
          {assetStats.length > 0 && <ExtremeLossTable assets={assetStats} />}

          {/* Regime Split */}
          {assetStats.length > 0 && <RegimeTable assets={assetStats} />}

          {/* Vol Regime */}
          {assetStats.length > 0 && <VolRegimeTable assets={assetStats} />}

          {/* Insights & Interpretation */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary" />
                <CardTitle className="text-sm font-semibold">Insights &amp; Interpretation</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">

              {/* ── 1. Normality Assessment ── */}
              <InsightRow
                title={port?.normal
                  ? 'Normality Assessment — Approximately Normal'
                  : 'Normality Assessment — Non-Normal Distribution'}
              >
                {port?.normal
                  ? `Jarque-Bera p-value = ${fmt(port.jbP)} — the null hypothesis of normality cannot be rejected at the 5% level. The return distribution is approximately bell-shaped, meaning skewness (${fmt(port.skewness, 2)}) and excess kurtosis (${fmt(port.kurtosis, 2)}) are statistically consistent with a normal distribution. Standard mean-variance risk models and parametric VaR are approximately valid for this portfolio.`
                  : `Jarque-Bera p-value = ${fmt(port?.jbP)} — normality is decisively rejected (p < 0.05). The distribution exhibits significant departures from the Gaussian assumption: skewness = ${fmt(port?.skewness, 2)}, excess kurtosis = ${fmt(port?.kurtosis, 2)}. Risk models built on the normality assumption — including parametric VaR, mean-variance optimisation, and standard Sharpe ratios — will systematically underestimate the true risk embedded in this return series.`}
              </InsightRow>

              {/* ── 2. Skewness ── */}
              <InsightRow
                title={
                  (port?.skewness ?? 0) < -0.5
                    ? 'Skewness — Negative (Left Tail Risk)'
                    : (port?.skewness ?? 0) > 0.5
                    ? 'Skewness — Positive (Right Tail Opportunity)'
                    : 'Skewness — Approximately Symmetric'
                }
              >
                {(port?.skewness ?? 0) < -0.5
                  ? `Skewness = ${fmt(port?.skewness, 2)}. The distribution has a heavy left tail — large losses occur more frequently and with greater magnitude than large gains. This is the most common and dangerous form of return asymmetry in equity portfolios. Even if the average return looks acceptable, the portfolio is exposed to periodic sharp drawdowns that are structurally larger than the upside. A negatively skewed portfolio requires more active risk management and position sizing discipline than volatility metrics alone suggest.`
                  : (port?.skewness ?? 0) > 0.5
                  ? `Skewness = ${fmt(port?.skewness, 2)}. The distribution has a right-skewed structure — large positive returns occur more frequently than symmetry would predict. While this is generally favourable, it can also reflect a lottery-like payoff structure where the average is pulled up by rare outlier gains. Verify whether the right-tail events are repeatable or driven by one-off events that may not persist.`
                  : `Skewness = ${fmt(port?.skewness, 2)}. The return distribution is approximately symmetric around the mean. Gains and losses of similar magnitude occur with roughly equal frequency. This is consistent with a well-balanced risk profile, though it does not eliminate the possibility of fat tails — check kurtosis separately.`}
              </InsightRow>

              {/* ── 3. Kurtosis / Fat Tails ── */}
              <InsightRow
                title={
                  (port?.kurtosis ?? 0) > 3
                    ? 'Kurtosis — Severe Fat Tails (Leptokurtic)'
                    : (port?.kurtosis ?? 0) > 1
                    ? 'Kurtosis — Fat Tails Detected'
                    : (port?.kurtosis ?? 0) > 0
                    ? 'Kurtosis — Mildly Fat Tails'
                    : 'Kurtosis — Thin Tails (Platykurtic)'
                }
              >
                {(port?.kurtosis ?? 0) > 3
                  ? `Excess kurtosis = ${fmt(port?.kurtosis, 2)} — extreme fat tails. Returns of 3–5 standard deviations from the mean occur far more frequently than the normal distribution predicts. This is the statistical signature of "black swan" risk: infrequent but catastrophic events that are invisible in standard volatility measures. Standard VaR will dramatically underestimate tail risk. CVaR (Expected Shortfall) and stress scenario analysis are essential complements to any volatility-based metric for this portfolio.`
                  : (port?.kurtosis ?? 0) > 1
                  ? `Excess kurtosis = ${fmt(port?.kurtosis, 2)} — meaningful fat tails. Extreme return events occur more frequently than normal. Drawdowns will tend to be larger and more abrupt than standard deviation-based models predict. This is typical of equity and credit return series, and should be explicitly accounted for in position sizing and risk budgeting.`
                  : (port?.kurtosis ?? 0) > 0
                  ? `Excess kurtosis = ${fmt(port?.kurtosis, 2)} — mild fat tails. The distribution has slightly heavier tails than normal, but the deviation is not large enough to fundamentally invalidate standard risk models. Monitor for any increase in kurtosis over time as a leading indicator of tail risk build-up.`
                  : `Excess kurtosis = ${fmt(port?.kurtosis, 2)} — thin tails. The distribution has lighter tails than a normal distribution, meaning extreme events are less frequent than expected. This is unusual for financial return series and may reflect a short sample period or smoothed pricing.`}
              </InsightRow>

              {/* ── 4. Tail Risk & VaR ── */}
              <InsightRow title="Tail Risk — Left Tail &amp; VaR Interpretation">
                {port
                  ? `The left tail of the distribution represents the loss region. Empirical tail probabilities: P(r &lt; −1%) = ${pct(port.tailProbs?.lt_1pct)}, P(r &lt; −2%) = ${pct(port.tailProbs?.lt_2pct)}, P(r &lt; −5%) = ${pct(port.tailProbs?.lt_5pct)}. ${
                      (port.tailProbs?.lt_5pct ?? 0) > 0.1
                        ? 'The probability of a −5% or worse period is elevated — the portfolio has a structurally heavy left tail that warrants active drawdown management and stop-loss discipline.'
                        : (port.tailProbs?.lt_5pct ?? 0) > 0.05
                        ? 'Tail loss probabilities are moderate. Standard VaR may be adequate but should be supplemented with CVaR for a complete picture of downside exposure.'
                        : 'Tail loss probabilities are within normal bounds. The left tail is not a primary risk concern at current levels, but this should be monitored during periods of rising volatility.'
                    } Tail ratio = ${port.tailRatio != null ? port.tailRatio.toFixed(2) : '—'} (right tail / left tail magnitude — above 1.0 is favourable).`
                  : 'Tail risk data unavailable.'}
              </InsightRow>

              {/* ── 5. Loss Asymmetry ── */}
              <InsightRow
                title={
                  (port?.gainLossRatio ?? 1) < 0.8
                    ? 'Loss Asymmetry — Losses Exceed Gains'
                    : (port?.gainLossRatio ?? 1) > 1.2
                    ? 'Loss Asymmetry — Gains Exceed Losses'
                    : 'Loss Asymmetry — Balanced Gain/Loss Profile'
                }
              >
                {port
                  ? `Average gain per positive period: ${signed(port.avgGain)} · Average loss per negative period: ${signed(port.avgLoss)} · Gain/Loss ratio: ${port.gainLossRatio != null ? port.gainLossRatio.toFixed(2) : '—'}. ${
                      (port.gainLossRatio ?? 1) < 0.8
                        ? 'Losses are materially larger than gains in absolute terms. Even if the portfolio has a high win rate, the negative edge on losing periods can erode compounding returns over time. This profile calls for tight loss management — letting losses run while cutting winners early is the most common way this ratio deteriorates.'
                        : (port.gainLossRatio ?? 1) > 1.2
                        ? 'Gains systematically exceed losses in magnitude — a positive asymmetric payoff structure. This is the hallmark of a well-managed portfolio or a momentum/trend-following regime. The key risk is a regime change that compresses this ratio during drawdown periods.'
                        : 'Gains and losses are roughly balanced in magnitude. The portfolio\'s long-term return will be primarily determined by win rate rather than edge size. Focus on consistency of positive periods rather than maximising individual return magnitude.'
                    }`
                  : 'Gain/loss data unavailable.'}
              </InsightRow>

              {/* ── 6. Regime Analysis ── */}
              {assetStats.length > 0 && (() => {
                const worstSkewAsset = [...assetStats].sort((a, b) => a.skewness - b.skewness)[0];
                const fatTailAssets  = assetStats.filter(a => a.kurtosis > 1);
                const nonNormalAssets = assetStats.filter(a => !a.normal);
                return (
                  <InsightRow title="Per-Asset Distribution — Key Findings">
                    {nonNormalAssets.length > 0
                      ? `${nonNormalAssets.length} of ${assetStats.length} asset${nonNormalAssets.length > 1 ? 's' : ''} (${nonNormalAssets.map(a => a.ticker).join(', ')}) fail the Jarque-Bera normality test individually. `
                      : `All ${assetStats.length} assets pass the Jarque-Bera normality test individually. `}
                    {fatTailAssets.length > 0
                      ? `Fat tails (excess kurtosis > 1) detected in: ${fatTailAssets.map(a => `${a.ticker} (κ = ${a.kurtosis.toFixed(2)})`).join(', ')}. `
                      : 'No individual assets show significant fat tails. '}
                    {worstSkewAsset && worstSkewAsset.skewness < -0.5
                      ? `Most negatively skewed asset: ${worstSkewAsset.ticker} (skew = ${worstSkewAsset.skewness.toFixed(2)}) — this name carries the highest loss asymmetry risk in the portfolio and warrants individual position size scrutiny.`
                      : worstSkewAsset && worstSkewAsset.skewness > 0.5
                      ? `Most positively skewed asset: ${[...assetStats].sort((a,b) => b.skewness - a.skewness)[0].ticker} (skew = ${[...assetStats].sort((a,b) => b.skewness - a.skewness)[0].skewness.toFixed(2)}) — right-skewed distribution, gains dominate over losses in magnitude.`
                      : 'No individual asset shows extreme skewness (all within ±0.5).'}
                  </InsightRow>
                );
              })()}

              {/* ── 7. Log scale note ── */}
              {logScale && (
                <InsightRow title="Log Return Scale Active">
                  Results are computed on log-transformed returns (ln(1 + r)). Log returns are additive over time and compress the impact of outliers, which can make distributions appear more symmetric. If skewness and kurtosis appear more benign in log scale than in simple returns, the difference represents the non-linear effect of compounding — the simple return distribution remains the more relevant measure for single-period risk assessment.
                </InsightRow>
              )}

              {/* ── Methodology note ── */}
              <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/40 border border-border">
                <Info className="w-3.5 h-3.5 text-muted-foreground shrink-0 mt-0.5" />
                <p className="text-xs text-muted-foreground">
                  Normality test: Jarque-Bera (H₀: skewness = 0 and excess kurtosis = 0). Reject at p &lt; 0.05.
                  Skewness: negative = left tail (asymmetric loss risk). Excess kurtosis &gt; 0 = fat tails vs normal.
                  Tail ratio = 95th pct / 5th pct absolute magnitude. Q-Q plot: deviation from diagonal = departure from normality.
                  Downside deviation uses 0% MAR threshold.
                </p>
              </div>

              {/* ── Warnings ── */}
              <div className="space-y-2 pt-2 border-t border-border">
                <p className="text-sm font-semibold text-foreground">Warnings</p>

                {/* Conditional: normality rejected (p < 0.05) */}
                {!port?.normal && (
                  <div className="flex gap-3 items-start p-3 rounded-lg border border-amber-200 bg-amber-500/5">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 bg-amber-500/10 text-amber-600 border-amber-200">High</span>
                    <p className="text-sm text-foreground leading-relaxed">
                      <strong>Normality Assumption Failure:</strong> The majority of financial risk models — including parametric VaR, Black-Scholes options pricing, mean-variance optimisation, and standard Sharpe ratio benchmarking — assume returns are normally distributed. Real equity return series almost universally violate this assumption. Using these models without adjustment for observed skewness and kurtosis will systematically underestimate portfolio risk, particularly during market stress periods.
                    </p>
                  </div>
                )}

                {/* Conditional: non-normal + fat tail */}
                {!port?.normal && (port?.kurtosis ?? 0) > 1 && (
                  <div className="flex gap-3 items-start p-3 rounded-lg border border-amber-200 bg-amber-500/5">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 bg-amber-500/10 text-amber-600 border-amber-200">High</span>
                    <p className="text-sm text-foreground leading-relaxed">
                      <strong>Hidden Tail Risk — VaR Underestimation:</strong> This portfolio is both non-normal and fat-tailed (excess kurtosis = {fmt(port?.kurtosis, 2)}). Standard 95% or 99% VaR calculated using a normal distribution assumption will significantly understate the true probability and magnitude of extreme losses. Use empirical (historical simulation) VaR or CVaR (Expected Shortfall) as the primary tail risk metric. Consider adding a kurtosis adjustment factor to any parametric risk models in use.
                    </p>
                  </div>
                )}

                {/* Conditional: negative skew */}
                {(port?.skewness ?? 0) < -0.5 && (
                  <div className="flex gap-3 items-start p-3 rounded-lg border border-amber-200 bg-amber-500/5">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 bg-amber-500/10 text-amber-600 border-amber-200">Medium</span>
                    <p className="text-sm text-foreground leading-relaxed">
                      <strong>Negative Skew Risk — Loss Asymmetry:</strong> Skewness = {fmt(port?.skewness, 2)}. The portfolio is structurally exposed to larger downside events than upside. This is often called "picking up pennies in front of a steamroller" — strategies that appear to have strong Sharpe ratios can carry hidden negative skew that only reveals itself during tail events. Standard Sharpe ratio overstates risk-adjusted performance for negatively skewed portfolios. Use skewness-adjusted Sharpe (Sortino or Omega ratio) for a more accurate assessment.
                    </p>
                  </div>
                )}

                {/* Always-on: Short sample bias */}
                {Math.min(...assets.map(a => a.returns?.length ?? 0)) < 60 && (
                  <div className="flex gap-3 items-start p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 bg-amber-500/10 text-amber-600 border-amber-200">Medium</span>
                    <p className="text-sm text-foreground leading-relaxed">
                      <strong>Short Sample Bias:</strong> The return series has fewer than 60 observations. Skewness and kurtosis estimates are highly sensitive to sample size — with small samples, a single outlier observation can dramatically shift both statistics. Treat the distribution shape as directional guidance rather than a precise measurement. Jarque-Bera test power is also substantially reduced at small sample sizes, meaning the test may fail to detect real non-normality.
                    </p>
                  </div>
                )}

                {/* Conditional: log scale distortion */}
                {logScale && (
                  <div className="flex gap-3 items-start p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 bg-green-500/10 text-green-600 border-green-200">Low</span>
                    <p className="text-sm text-foreground leading-relaxed">
                      <strong>Log Return Scale Effect:</strong> Log transformation compresses the impact of large returns and can make distributions appear more symmetric and normal than the underlying simple return series. Statistics displayed in log scale will generally show lower skewness and kurtosis than simple returns. For single-period risk assessment and position sizing, simple returns are the more relevant measure. Switch to simple return mode to see the uncompressed distribution shape.
                    </p>
                  </div>
                )}
              </div>

              {/* ── Recommended Actions ── */}
              {(() => {
                const actions: { priority: 'high' | 'medium' | 'low'; text: string }[] = [];

                // Non-normal → don't use standard VaR
                if (!port?.normal) {
                  actions.push({
                    priority: 'high',
                    text: `The portfolio fails the normality test (JB p = ${fmt(port?.jbP)}). Do not rely on parametric VaR as the sole risk metric. Switch to empirical (historical simulation) VaR or CVaR (Expected Shortfall), which make no distributional assumptions. If you use a mean-variance optimiser, apply a penalty for non-normality or switch to a downside-risk-based objective function such as Mean-CVaR optimisation.`,
                  });
                }

                // Fat tails → CVaR
                if ((port?.kurtosis ?? 0) > 1) {
                  actions.push({
                    priority: 'high',
                    text: `Excess kurtosis = ${fmt(port?.kurtosis, 2)} — fat tails are present. Add CVaR (Conditional VaR / Expected Shortfall) to your risk dashboard. CVaR measures the average loss in the worst X% of scenarios and directly captures fat-tail severity, unlike VaR which only marks the threshold. Run stress scenarios that assume 3–5 standard deviation return events, which this distribution suggests are materially more probable than the normal assumption implies.`,
                  });
                }

                // Negative skew → skew-adjusted metrics
                if ((port?.skewness ?? 0) < -0.5) {
                  actions.push({
                    priority: 'medium',
                    text: `Skewness = ${fmt(port?.skewness, 2)} — negative skew indicates asymmetric loss risk. Replace or supplement the Sharpe ratio with the Sortino ratio (which penalises only downside volatility) or the Omega ratio (which accounts for the full distribution shape). For position sizing, apply a skewness penalty: reduce position sizes in assets that individually contribute the most to portfolio negative skew.`,
                  });
                }

                // High tail probability
                if ((port?.tailProbs?.lt_5pct ?? 0) > 0.08) {
                  actions.push({
                    priority: 'medium',
                    text: `Empirical P(r < −5%) = ${pct(port?.tailProbs?.lt_5pct)} — the probability of a severe loss period is elevated. Consider implementing a systematic drawdown limit that triggers position reduction when rolling losses approach the 5th percentile threshold. Additionally, evaluate whether options-based tail hedging (e.g., OTM put spreads) is cost-effective given the observed tail probability.`,
                  });
                }

                // Loss asymmetry
                if ((port?.gainLossRatio ?? 1) < 0.85) {
                  actions.push({
                    priority: 'medium',
                    text: `Gain/Loss ratio = ${port?.gainLossRatio?.toFixed(2) ?? '—'} — losses exceed gains in magnitude. Review the portfolio for positions with asymmetric loss structures. Common causes include overweight positions in high-beta assets, implicit short-volatility exposures, or concentrated sector bets. Rebalancing toward assets with more symmetric return profiles (higher gain/loss ratio) will improve the compounding profile even if it reduces headline return.`,
                  });
                }

                // Non-normal individual assets
                const nonNormalCount = assetStats.filter(a => !a.normal).length;
                if (nonNormalCount > 0 && assetStats.length > 1) {
                  actions.push({
                    priority: 'low',
                    text: `${nonNormalCount} individual asset${nonNormalCount > 1 ? 's' : ''} (${assetStats.filter(a => !a.normal).map(a => a.ticker).join(', ')}) fail the normality test. When combining these into a portfolio, the non-normality may partially cancel (diversification of skewness) or compound (correlated tail events). Check the correlation matrix page to assess whether the non-normal assets tend to move together during their extreme return periods — correlated tails are significantly more dangerous than independent tails.`,
                  });
                }

                // Well-behaved distribution
                if (port?.normal && (port?.kurtosis ?? 0) <= 0.5 && Math.abs(port?.skewness ?? 0) <= 0.3) {
                  actions.push({
                    priority: 'low',
                    text: `The portfolio return distribution is well-behaved — approximately normal, low excess kurtosis, and symmetric. Standard risk models are valid for this portfolio at current composition. Continue monitoring distribution statistics after any significant portfolio changes, as adding new assets or changing weights can materially alter the aggregate distribution shape.`,
                  });
                }

                if (!actions.length) {
                  actions.push({
                    priority: 'low',
                    text: 'Distribution statistics are within normal bounds. No immediate model adjustments required. Continue monitoring skewness, kurtosis, and normality test results on a rolling basis as market regimes and portfolio composition evolve.',
                  });
                }

                const meta = {
                  high:   { label: 'High',   cls: 'bg-red-500/10 text-red-600 border-red-200' },
                  medium: { label: 'Medium', cls: 'bg-amber-500/10 text-amber-600 border-amber-200' },
                  low:    { label: 'Low',    cls: 'bg-green-500/10 text-green-600 border-green-200' },
                };
                return (
                  <div className="space-y-2 pt-2 border-t border-border">
                    <p className="text-sm font-semibold text-foreground">Recommended Actions</p>
                    {actions.map((a, i) => (
                      <div key={i} className="flex gap-3 items-start p-3 rounded-lg border border-border bg-muted/20">
                        <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shrink-0 mt-0.5 ${meta[a.priority].cls}`}>
                          {meta[a.priority].label}
                        </span>
                        <p className="text-sm text-foreground leading-relaxed">{a.text}</p>
                      </div>
                    ))}
                  </div>
                );
              })()}

            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
