import type { Prediction } from "@/ui/types";
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Bar, BarChart } from "recharts";

interface GoalsDistributionProps {
  match: Prediction;
}

const CHART_COLORS = {
  primary: "#34d399",
  secondary: "#818cf8",
  text: "#a1a1aa",
  tooltipBg: "#27272a",
  tooltipBorder: "#3f3f46",
};

export function GoalsDistribution({ match }: GoalsDistributionProps) {
  const dist: any = match.distribution ?? {};
  const expectedHome = typeof dist.expected_goals_home === "number" ? dist.expected_goals_home : 0;
  const expectedAway = typeof dist.expected_goals_away === "number" ? dist.expected_goals_away : 0;
  const p_btts = typeof dist.p_btts === "number" ? dist.p_btts : 0;
  const p_over_0_5 = typeof dist.p_over_0_5 === "number" ? dist.p_over_0_5 : 0;
  const p_over_1_5 = typeof dist.p_over_1_5 === "number" ? dist.p_over_1_5 : 0;
  const p_over_2_5 = typeof dist.p_over_2_5 === "number" ? dist.p_over_2_5 : 0;
  const p_over_3_5 = typeof dist.p_over_3_5 === "number" ? dist.p_over_3_5 : 0;

  const overUnderData = [
    { threshold: "O 0.5", probability: p_over_0_5 * 100 },
    { threshold: "O 1.5", probability: p_over_1_5 * 100 },
    { threshold: "O 2.5", probability: p_over_2_5 * 100 },
    { threshold: "O 3.5", probability: p_over_3_5 * 100 },
  ];

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">Goals Distribution</div>
      </div>
      <div className="p-6 pt-2">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-secondary p-3">
                <p className="text-xs text-muted-foreground">Expected (Home)</p>
                <p className="text-2xl font-bold text-emerald-400">{expectedHome.toFixed(2)}</p>
              </div>
              <div className="rounded-lg bg-secondary p-3">
                <p className="text-xs text-muted-foreground">Expected (Away)</p>
                <p className="text-2xl font-bold text-indigo-400">{expectedAway.toFixed(2)}</p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">BTTS Probability</span>
                <span className="font-medium text-foreground">{(p_btts * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full bg-emerald-400 rounded-full transition-all"
                  style={{ width: `${p_btts * 100}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-secondary/50 p-2">
                <p className="text-xs text-muted-foreground">Median Goals</p>
                <p className="font-semibold text-foreground">{dist.total_goals_median ?? "—"}</p>
              </div>
              <div className="rounded-lg bg-secondary/50 p-2">
                <p className="text-xs text-muted-foreground">IQR</p>
                <p className="font-semibold text-foreground">{dist.total_goals_q25_q75 ?? "—"}</p>
              </div>
            </div>
          </div>

          <div>
            <p className="text-sm text-muted-foreground mb-2">Over/Under Probabilities</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={overUnderData}>
                <XAxis dataKey="threshold" stroke={CHART_COLORS.text} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  stroke={CHART_COLORS.text}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Probability"]}
                  labelStyle={{ color: "#fff" }}
                />
                <Bar dataKey="probability" fill={CHART_COLORS.primary} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
