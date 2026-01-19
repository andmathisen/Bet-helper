import type { Prediction } from "@/ui/types";
import { Bar, BarChart, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface EVAnalysisProps {
  match: Prediction;
  evModel: "poisson" | "ml";
}

const CHART_COLORS = {
  positive: "#34d399",
  negative: "#f87171",
  text: "#a1a1aa",
  reference: "#71717a",
  tooltipBg: "#27272a",
  tooltipBorder: "#3f3f46",
};

export function EVAnalysis({ match, evModel }: EVAnalysisProps) {
  const ev = evModel === "ml" ? match.ml_ev : match.ev;
  const evData = ev ?? { home: null, draw: null, away: null };
  
  const data = [
    { outcome: "Home", ev: (evData.home ?? 0) * 100, isPositive: (evData.home ?? 0) > 0 },
    { outcome: "Draw", ev: (evData.draw ?? 0) * 100, isPositive: (evData.draw ?? 0) > 0 },
    { outcome: "Away", ev: (evData.away ?? 0) * 100, isPositive: (evData.away ?? 0) > 0 },
  ];

  const hasPositiveEV = data.some((d) => d.isPositive);
  const bestBet = data.reduce((best, current) => (current.ev > best.ev ? current : best));

  const evValues = data.map((d) => d.ev);
  const avgEV = evValues.reduce((a, b) => a + b, 0) / evValues.length;
  const stdDev = Math.sqrt(evValues.reduce((sum, val) => sum + Math.pow(val - avgEV, 2), 0) / evValues.length);
  const outliers = data.filter((d) => Math.abs(d.ev - avgEV) > stdDev * 1.5);

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold flex items-center gap-2">
          Expected Value Analysis ({evModel === "ml" ? "ML Model" : "Poisson"})
          {hasPositiveEV ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-red-400" />
          )}
        </div>
      </div>
      <div className="p-6 pt-2">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} layout="vertical">
            <XAxis
              type="number"
              stroke={CHART_COLORS.text}
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value.toFixed(1)}%`}
              domain={["dataMin - 5", "dataMax + 5"]}
            />
            <YAxis
              type="category"
              dataKey="outcome"
              stroke={CHART_COLORS.text}
              fontSize={12}
              tickLine={false}
              axisLine={false}
              width={50}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: CHART_COLORS.tooltipBg,
                border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                borderRadius: "8px",
                color: "#fff",
              }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, "EV"]}
              labelStyle={{ color: "#fff" }}
            />
            <ReferenceLine x={0} stroke={CHART_COLORS.reference} strokeDasharray="3 3" />
            <Bar dataKey="ev" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.isPositive ? CHART_COLORS.positive : CHART_COLORS.negative} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Best Bet:</span>
            <span className={`font-medium ${bestBet.isPositive ? "text-emerald-400" : "text-muted-foreground"}`}>
              {bestBet.outcome} ({bestBet.ev.toFixed(2)}%)
            </span>
          </div>

          {outliers.length > 0 && (
            <div className="flex items-start gap-2 rounded-lg bg-secondary/50 p-3 text-sm">
              <AlertTriangle className="h-4 w-4 text-amber-400 mt-0.5 shrink-0" />
              <div>
                <span className="font-medium text-amber-400">Outlier detected: </span>
                <span className="text-muted-foreground">
                  {outliers.map((o) => `${o.outcome} (${o.ev.toFixed(2)}%)`).join(", ")} deviates significantly from
                  average
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
