import type { Prediction } from "@/ui/types";
import { Bar, BarChart, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend } from "recharts";
import { maxEv } from "@/ui/utils";

interface MatchComparisonProps {
  matches: Prediction[];
}

const CHART_COLORS = {
  positive: "#34d399",
  negative: "#f87171",
  home: "#34d399",
  away: "#818cf8",
  text: "#a1a1aa",
  tooltipBg: "#27272a",
  tooltipBorder: "#3f3f46",
};

export function MatchComparison({ matches }: MatchComparisonProps) {
  const evData = matches.map((m) => {
    const best = maxEv(m);
    return {
      match: `${m.match.home.split(" ")[0]} vs ${m.match.away.split(" ")[0]}`,
      "Best EV": (best?.ev ?? 0) * 100,
      recommended: m.recommended.outcome,
    };
  });

  const goalsData = matches.map((m) => {
    const dist: any = m.distribution ?? {};
    return {
      match: `${m.match.home.split(" ")[0]} vs ${m.match.away.split(" ")[0]}`,
      "Home xG": typeof dist.expected_goals_home === "number" ? dist.expected_goals_home : 0,
      "Away xG": typeof dist.expected_goals_away === "number" ? dist.expected_goals_away : 0,
    };
  });

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">Match Comparison</div>
      </div>
      <div className="p-6 pt-2 space-y-6">
        <div>
          <h4 className="text-sm font-medium text-muted-foreground mb-3">Best EV by Match</h4>
          <ResponsiveContainer width="100%" height={Math.max(400, matches.length * 40)}>
            <BarChart data={evData} layout="vertical" margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
              <XAxis
                type="number"
                stroke={CHART_COLORS.text}
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `${v.toFixed(1)}%`}
              />
              <YAxis
                type="category"
                dataKey="match"
                stroke={CHART_COLORS.text}
                fontSize={12}
                tickLine={false}
                axisLine={false}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.tooltipBg,
                  border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                  borderRadius: "8px",
                  color: "#fff",
                }}
                formatter={(value: number, name: string, entry: any) => [
                  `${value.toFixed(2)}% (${entry.payload.recommended})`,
                  "Best EV",
                ]}
                labelStyle={{ color: "#fff" }}
              />
              <Bar dataKey="Best EV" radius={[0, 4, 4, 0]}>
                {evData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry["Best EV"] > 0 ? CHART_COLORS.positive : CHART_COLORS.negative}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h4 className="text-sm font-medium text-muted-foreground mb-3">Expected Goals Comparison</h4>
          <ResponsiveContainer width="100%" height={Math.max(400, matches.length * 40)}>
            <BarChart data={goalsData} margin={{ left: 10, right: 10, top: 10, bottom: 60 }}>
              <XAxis 
                dataKey="match" 
                stroke={CHART_COLORS.text} 
                fontSize={11} 
                tickLine={false} 
                axisLine={false}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis stroke={CHART_COLORS.text} fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.tooltipBg,
                  border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                  borderRadius: "8px",
                  color: "#fff",
                }}
                labelStyle={{ color: "#fff" }}
              />
              <Legend wrapperStyle={{ color: "#fff" }} />
              <Bar dataKey="Home xG" fill={CHART_COLORS.home} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Away xG" fill={CHART_COLORS.away} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
