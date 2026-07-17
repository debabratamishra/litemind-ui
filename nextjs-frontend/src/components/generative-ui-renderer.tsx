'use client';

import * as React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Info,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { parseGenerativeUI } from '@/lib/generative-ui';
import MarkdownRenderer from '@/components/markdown-renderer';

// ─── Colour palette for charts ──────────────────────────────────────────────

const CHART_COLORS = [
  'hsl(var(--chart-1, 221 83% 53%))',
  'hsl(var(--chart-2, 142 76% 36%))',
  'hsl(var(--chart-3, 43 96% 56%))',
  'hsl(var(--chart-4, 355 78% 56%))',
  'hsl(var(--chart-5, 262 83% 58%))',
];

// ─── Data shapes (must match the backend generative-UI schema in chat.py) ──

interface ChartData {
  type: 'bar' | 'line' | 'area' | 'pie' | 'scatter';
  title?: string;
  x: string[];
  y: number[];
}

interface TableData {
  title?: string;
  columns: string[];
  data: string[][];
}

interface MetricItem {
  label: string;
  value: string;
  delta?: string;
}

interface ButtonItem {
  text: string;
  value: string;
}

interface ButtonGroupData {
  label?: string;
  buttons: ButtonItem[];
  /** Legacy flat shape, kept for safety. */
  action?: string;
  payload?: string;
}

type AlertLevel = 'info' | 'success' | 'warning' | 'error';

interface InfoCardData {
  icon?: string;
  title?: string;
  content?: string;
  color?: string;
}

interface StepsData {
  steps: string[];
  current?: number;
}

interface TabItem {
  label: string;
  content: string;
}

interface CalloutData {
  emoji?: string;
  title?: string;
  content?: string;
}

interface ColumnItem {
  title?: string;
  content?: string;
  icon?: string;
}

interface JsonViewerData {
  title?: string;
  data: unknown;
}

// ─── Charts ────────────────────────────────────────────────────────────────

const tooltipStyle = {
  backgroundColor: 'hsl(var(--card))',
  border: '1px solid hsl(var(--border))',
  borderRadius: '8px',
  color: 'hsl(var(--foreground))',
  fontSize: '12px',
} as const;

function ChartRenderer({ data }: { data: ChartData }) {
  const { type, title, x = [], y = [] } = data;
  const chartData = x.map((xv, i) => ({ x: xv, y: y[i] ?? 0 }));

  const renderChart = () => {
    const common = {
      data: chartData,
      margin: { top: 8, right: 16, bottom: 8, left: 0 },
    };

    switch (type) {
      case 'bar':
        return (
          <BarChart {...common}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis dataKey="x" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <RechartsTooltip contentStyle={tooltipStyle} />
            <Legend />
            <Bar dataKey="y" fill={CHART_COLORS[0]} radius={[4, 4, 0, 0]} />
          </BarChart>
        );
      case 'line':
        return (
          <LineChart {...common}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis dataKey="x" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <RechartsTooltip contentStyle={tooltipStyle} />
            <Legend />
            <Line
              type="monotone"
              dataKey="y"
              stroke={CHART_COLORS[0]}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        );
      case 'area':
        return (
          <AreaChart {...common}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis dataKey="x" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <RechartsTooltip contentStyle={tooltipStyle} />
            <Legend />
            <Area
              type="monotone"
              dataKey="y"
              stroke={CHART_COLORS[0]}
              fill={CHART_COLORS[0]}
              fillOpacity={0.2}
              strokeWidth={2}
            />
          </AreaChart>
        );
      case 'scatter':
        return (
          <ScatterChart {...common}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis dataKey="x" type="number" tick={{ fontSize: 12 }} name="x" />
            <YAxis dataKey="y" type="number" tick={{ fontSize: 12 }} name="y" />
            <ZAxis range={[100, 100]} />
            <RechartsTooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={chartData} fill={CHART_COLORS[0]} />
          </ScatterChart>
        );
      case 'pie':
        return (
          <PieChart>
            <Pie
              data={chartData}
              dataKey="y"
              nameKey="x"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ name, percent }) =>
                `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`
              }
            >
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <RechartsTooltip contentStyle={tooltipStyle} />
            <Legend />
          </PieChart>
        );
      default:
        return null;
    }
  };

  return (
    <Card className="my-4">
      {title && (
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={280}>
          {renderChart() ?? <div />}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// ─── Table ──────────────────────────────────────────────────────────────────

function TableRenderer({ data }: { data: TableData }) {
  const { title, columns = [], data: rows = [] } = data;

  return (
    <Card className="my-4">
      {title && (
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-muted/50">
              <tr>
                {columns.map((header, i) => (
                  <th
                    key={i}
                    className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row, i) => (
                <tr key={i} className="hover:bg-muted/20 transition-colors">
                  {row.map((cell, j) => (
                    <td key={j} className="px-4 py-2.5 text-foreground">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Metric ────────────────────────────────────────────────────────────────

function MetricRenderer({ metrics }: { metrics: MetricItem[] }) {
  return (
    <div className="my-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
      {metrics.map((m, i) => {
        const trend =
          m.delta?.startsWith('-')
            ? 'down'
            : m.delta?.startsWith('+')
              ? 'up'
              : 'neutral';
        const TrendIcon =
          trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
        const trendColor =
          trend === 'up'
            ? 'text-green-500'
            : trend === 'down'
              ? 'text-red-500'
              : 'text-muted-foreground';

        return (
          <Card key={i}>
            <CardContent className="pt-6">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-muted-foreground">
                    {m.label}
                  </p>
                  <p className="mt-1 text-2xl font-bold text-foreground">
                    {m.value}
                  </p>
                  {m.delta && (
                    <p className={`mt-1 text-xs font-medium ${trendColor}`}>
                      {m.delta}
                    </p>
                  )}
                </div>
                {m.delta && (
                  <TrendIcon className={`h-4 w-4 shrink-0 ${trendColor}`} aria-hidden="true" />
                )}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

// ─── Button group ─────────────────────────────────────────────────────────

interface ButtonGroupProps {
  data: ButtonGroupData;
  onAction?: (action: string, payload?: string) => void;
}

function ButtonGroupRenderer({ data, onAction }: ButtonGroupProps) {
  const { label, buttons = [], action, payload } = data;
  const items: ButtonItem[] =
    buttons.length > 0
      ? buttons
      : action
        ? [{ text: label ?? 'Submit', value: payload ?? '' }]
        : [];

  return (
    <div className="my-3 space-y-2">
      {label && <p className="text-sm font-medium text-muted-foreground">{label}</p>}
      <div className="flex flex-wrap gap-2">
        {items.map((b, i) => (
          <Button
            key={i}
            variant="default"
            onClick={() => onAction?.('send_message', b.value)}
          >
            {b.text}
          </Button>
        ))}
      </div>
    </div>
  );
}

// ─── Progress ─────────────────────────────────────────────────────────────

function ProgressRenderer({ data }: { data: { label?: string; value?: number } }) {
  const pct = Math.min(100, Math.max(0, Number(data.value) || 0));
  return (
    <Card className="my-4">
      <CardContent className="pt-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-foreground">{data.label}</span>
            <Badge variant="secondary">{pct}%</Badge>
          </div>
          <Progress value={pct} className="h-2" />
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Info card ──────────────────────────────────────────────────────────

function InfoCardRenderer({ data }: { data: InfoCardData }) {
  const style = data.color ? { borderLeftColor: data.color, borderLeftWidth: 4 } : undefined;
  return (
    <Card className="my-4" style={style}>
      <CardContent className="space-y-1 pt-6">
        <div className="flex items-center gap-2">
          {data.icon && <span className="text-lg" aria-hidden="true">{data.icon}</span>}
          {data.title && (
            <p className="font-semibold text-foreground">{data.title}</p>
          )}
        </div>
        {data.content && (
          <p className="text-sm leading-relaxed text-muted-foreground">{data.content}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Alert ─────────────────────────────────────────────────────────────

function AlertRenderer({ data }: { data: { level: AlertLevel; message: string } }) {
  const map: Record<AlertLevel, { cls: string; Icon: typeof Info }> = {
    info: { cls: 'border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400', Icon: Info },
    success: { cls: 'border-green-500/30 bg-green-500/10 text-green-600 dark:text-green-400', Icon: CheckCircle2 },
    warning: { cls: 'border-yellow-500/30 bg-yellow-500/10 text-yellow-600 dark:text-yellow-400', Icon: AlertTriangle },
    error: { cls: 'border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-400', Icon: XCircle },
  };
  const { cls, Icon } = map[data.level] ?? map.info;
  return (
    <div className={`my-4 flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm ${cls}`} role="alert">
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>{data.message}</span>
    </div>
  );
}

// ─── Steps ─────────────────────────────────────────────────────────────

function StepsRenderer({ data }: { data: StepsData }) {
  const { steps = [], current } = data;
  return (
    <ol className="my-4 space-y-2">
      {steps.map((step, i) => {
        const isCurrent = typeof current === 'number' && current === i;
        const isDone = typeof current === 'number' && i < current;
        return (
          <li key={i} className="flex items-start gap-2.5">
            <span
              className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                isDone
                  ? 'bg-green-500 text-white'
                  : isCurrent
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground'
              }`}
            >
              {isDone ? <CheckCircle2 className="h-3.5 w-3.5" /> : i + 1}
            </span>
            <span
              className={`text-sm leading-relaxed ${
                isCurrent ? 'font-medium text-foreground' : 'text-muted-foreground'
              }`}
            >
              {step}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

// ─── Tabs ────────────────────────────────────────────────────────────────

function TabsRenderer({ data }: { data: { tabs?: TabItem[] } }) {
  const tabs = data.tabs ?? [];
  const [active, setActive] = React.useState(0);

  if (tabs.length === 0) return null;

  return (
    <div className="my-4 rounded-lg border border-border">
      <div className="flex flex-wrap gap-1 border-b border-border px-2 pt-2" role="tablist">
        {tabs.map((t, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === active}
            onClick={() => setActive(i)}
            className={`flex items-center gap-1 rounded-t-md px-3 py-1.5 text-sm transition-colors ${
              i === active
                ? 'bg-muted font-medium text-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="p-4" role="tabpanel">
        <MarkdownRenderer content={tabs[active]?.content ?? ''} />
      </div>
    </div>
  );
}

// ─── Callout ──────────────────────────────────────────────────────────

function CalloutRenderer({ data }: { data: CalloutData }) {
  return (
    <div className="my-4 flex items-start gap-2.5 rounded-lg border border-border bg-muted/40 px-3 py-2.5">
      {data.emoji && <span className="text-lg" aria-hidden="true">{data.emoji}</span>}
      <div className="space-y-1">
        {data.title && <p className="font-semibold text-foreground">{data.title}</p>}
        {data.content && (
          <p className="text-sm leading-relaxed text-muted-foreground">{data.content}</p>
        )}
      </div>
    </div>
  );
}

// ─── Columns ──────────────────────────────────────────────────────────

function ColumnsRenderer({ data }: { data: { items?: ColumnItem[] } }) {
  const items = data.items ?? [];
  return (
    <div className="my-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((item, i) => (
        <Card key={i}>
          <CardContent className="space-y-1 pt-6">
            <div className="flex items-center gap-2">
              {item.icon && <span className="text-lg" aria-hidden="true">{item.icon}</span>}
              {item.title && (
                <p className="font-semibold text-foreground">{item.title}</p>
              )}
            </div>
            {item.content && (
              <p className="text-sm leading-relaxed text-muted-foreground">{item.content}</p>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ─── JSON viewer ─────────────────────────────────────────────────────────

function JsonViewerRenderer({ data }: { data: JsonViewerData }) {
  let pretty = '';
  try {
    pretty = JSON.stringify(data.data, null, 2);
  } catch {
    pretty = String(data.data);
  }
  return (
    <Card className="my-4">
      {data.title && (
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">{data.title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <pre className="overflow-x-auto rounded-md bg-muted/50 p-4 font-mono text-xs leading-relaxed text-foreground">
          {pretty}
        </pre>
      </CardContent>
    </Card>
  );
}

// ─── HTML (html / webapp / iframe_app) ────────────────────────────────────
//
// Renders AI-generated self-contained HTML. The iframe is sandboxed with
// `allow-scripts` but WITHOUT `allow-same-origin`, so it gets an opaque
// origin and the parent page CANNOT read its DOM (which keeps the AI HTML
// from touching our cookies / localStorage / parent DOM). Because of that,
// we must not write/read the document from the parent. Instead the HTML is
// supplied declaratively via the `srcDoc` attribute, and an in-iframe script
// measures its OWN height and posts it to the parent over `postMessage`
// (the sanctioned cross-origin channel). The parent resizes to it, so the
// frame is never a fixed-size blank box.

interface HtmlRendererProps {
  html: string;
}

const HTML_HEIGHT_MSG = 'litemind-html-height';

function HtmlRenderer({ html }: HtmlRendererProps) {
  const id = React.useId();

  // Initial size: honor the backend's `<!-- height: N -->` hint if present.
  const [height, setHeight] = React.useState(() => {
    const m = /<!--\s*height:\s*(\d+)\s*-->/.exec(html);
    return m ? parseInt(m[1], 10) : 300;
  });

  const srcDoc = React.useMemo(() => `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          html, body { height: auto; }
          body {
            font-family: system-ui, -apple-system, sans-serif;
            padding: 12px;
            color: #e5e7eb;
            background: transparent;
          }
        </style>
        <script>
          (function () {
            function report() {
              var h = document.body ? document.body.scrollHeight : 0;
              if (h > 0) {
                parent.postMessage(
                  { type: ${JSON.stringify(HTML_HEIGHT_MSG)}, id: ${JSON.stringify(id)}, height: h },
                  '*'
                );
              }
            }
            window.addEventListener('load', report);
            setTimeout(report, 300);
            setTimeout(report, 1000);
            if (window.ResizeObserver) {
              new ResizeObserver(report).observe(document.body);
            }
            window.addEventListener('resize', report);
          })();
        <\/script>
      </head>
      <body>${html}</body>
    </html>
  `, [html, id]);

  // Listen for height reports from THIS sandbox (ignore other iframes / sources).
  React.useEffect(() => {
    function onMessage(e: MessageEvent) {
      const data = e.data as { type?: string; id?: string; height?: number } | null;
      if (
        data &&
        data.type === HTML_HEIGHT_MSG &&
        data.id === id &&
        typeof data.height === 'number' &&
        data.height > 0
      ) {
        setHeight(Math.ceil(data.height) + 16);
      }
    }
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [id]);

  return (
    <div className="my-4 overflow-hidden rounded-lg border border-border">
      <iframe
        srcDoc={srcDoc}
        sandbox="allow-scripts"
        title="Rendered HTML content"
        className="block w-full bg-transparent"
        style={{ height, border: 'none', display: 'block' }}
        scrolling="no"
      />
    </div>
  );
}

// ─── Main GenerativeUIRenderer ──────────────────────────────────────────

interface GenerativeUIRendererProps {
  content: string;
  onAction?: (action: string, payload?: string) => void;
}

export default function GenerativeUIRenderer({
  content,
  onAction,
}: GenerativeUIRendererProps) {
  const segments = React.useMemo(() => parseGenerativeUI(content), [content]);

  return (
    <div className="space-y-1">
      {segments.map((segment, index) => {
        if (segment.type === 'text') {
          return <MarkdownRenderer key={index} content={segment.content} />;
        }

        // UI segment — validate payload shape, fall back to raw markdown on mismatch.
        const d = segment.data as Record<string, unknown> | undefined;
        const fallback = () => (
          <MarkdownRenderer key={index} content={segment.content} />
        );

        switch (segment.component) {
          case 'chart': {
            const c = d as unknown as ChartData;
            if (!Array.isArray(c?.x) || !Array.isArray(c?.y) || c.x.length === 0) {
              return fallback();
            }
            return <ChartRenderer key={index} data={c} />;
          }
          case 'table': {
            const t = d as unknown as TableData;
            if (!Array.isArray(t?.columns) || !Array.isArray(t?.data)) {
              return fallback();
            }
            return <TableRenderer key={index} data={t} />;
          }
          case 'metric': {
            const m = d as { metrics?: MetricItem[] };
            if (!Array.isArray(m?.metrics) || m.metrics.length === 0) {
              return fallback();
            }
            return <MetricRenderer key={index} metrics={m.metrics} />;
          }
          case 'button':
          case 'button_group': {
            const b = d as unknown as ButtonGroupData;
            if (
              (!Array.isArray(b?.buttons) || b.buttons.length === 0) &&
              !b?.action
            ) {
              return fallback();
            }
            return <ButtonGroupRenderer key={index} data={b} onAction={onAction} />;
          }
          case 'progress': {
            const p = d as { label?: string; value?: number };
            if (typeof p?.value !== 'number') {
              return fallback();
            }
            return <ProgressRenderer key={index} data={p} />;
          }
          case 'info_card': {
            const ic = d as InfoCardData;
            if (!ic?.icon && !ic?.title && !ic?.content) return fallback();
            return <InfoCardRenderer key={index} data={ic} />;
          }
          case 'alert': {
            const a = d as { level?: AlertLevel; message?: string };
            if (!a?.message) return fallback();
            return (
              <AlertRenderer
                key={index}
                data={{ level: a.level ?? 'info', message: a.message }}
              />
            );
          }
          case 'steps': {
            const s = d as unknown as StepsData;
            if (!Array.isArray(s?.steps) || s.steps.length === 0) return fallback();
            return <StepsRenderer key={index} data={s} />;
          }
          case 'tabs': {
            const tb = d as { tabs?: TabItem[] };
            if (!Array.isArray(tb?.tabs) || tb.tabs.length === 0) return fallback();
            return <TabsRenderer key={index} data={tb} />;
          }
          case 'callout': {
            const co = d as CalloutData;
            if (!co?.emoji && !co?.title && !co?.content) return fallback();
            return <CalloutRenderer key={index} data={co} />;
          }
          case 'columns': {
            const col = d as { items?: ColumnItem[] };
            if (!Array.isArray(col?.items) || col.items.length === 0) return fallback();
            return <ColumnsRenderer key={index} data={col} />;
          }
          case 'json_viewer': {
            const j = d as unknown as JsonViewerData;
            if (j?.data === undefined) return fallback();
            return <JsonViewerRenderer key={index} data={j} />;
          }
          case 'html':
          case 'webapp':
          case 'iframe_app': {
            return <HtmlRenderer key={index} html={segment.content} />;
          }
          default:
            return fallback();
        }
      })}
    </div>
  );
}
