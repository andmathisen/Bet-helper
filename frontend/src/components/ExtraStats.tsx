import type { Prediction } from "@/ui/types";
import { Bar, BarChart, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { CornerDownRight, CreditCard, Target, Crosshair } from "lucide-react";

const CHART_COLORS = {
  chart1: "#34d399", // emerald-400
  chart2: "#818cf8", // indigo-400
  chart3: "#fbbf24", // amber-400
  chart4: "#ec4899", // pink-500
  text: "#a1a1aa", // zinc-400
  tooltipBg: "#27272a", // zinc-800
  tooltipBorder: "#3f3f46", // zinc-700
};
import * as Tabs from "@radix-ui/react-tabs";

interface ExtraStatsProps {
  match: Prediction;
}

function StatOverview({
  label,
  expected,
  median,
  iqr,
  icon: Icon,
}: {
  label: string;
  expected: string;
  median: number | null;
  iqr: string | null;
  icon: React.ElementType;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-secondary p-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
        <Icon className="h-5 w-5 text-primary" />
      </div>
      <div className="flex-1">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground">{expected ?? "—"}</p>
      </div>
      <div className="text-right">
        <p className="text-lg font-bold text-foreground">{median ?? "—"}</p>
        <p className="text-xs text-muted-foreground">IQR: {iqr ?? "—"}</p>
      </div>
    </div>
  );
}

export function ExtraStats({ match }: ExtraStatsProps) {
  const extra_stats: any = match.extra_stats ?? {};

  const corners: any = extra_stats.corners ?? {};
  const cards: any = extra_stats.cards ?? {};
  const shots: any = extra_stats.shots ?? {};
  const sot: any = extra_stats.sot ?? {};

  const cornersData = Object.entries(corners)
    .filter(([k]) => k.startsWith("p_over_") && k.endsWith("_total"))
    .map(([k, v]) => {
      const line = k.replace("p_over_", "").replace("_total", "");
      return {
        threshold: `O ${line}`,
        probability: (typeof v === "number" ? v : 0) * 100,
      };
    })
    .slice(0, 4);

  const cardsData = Object.entries(cards)
    .filter(([k]) => k.startsWith("p_over_") && k.endsWith("_total"))
    .map(([k, v]) => {
      const line = k.replace("p_over_", "").replace("_total", "");
      return {
        threshold: `O ${line}`,
        probability: (typeof v === "number" ? v : 0) * 100,
      };
    })
    .slice(0, 4);

  const shotsData = Object.entries(shots)
    .filter(([k]) => k.startsWith("p_over_") && k.endsWith("_total"))
    .map(([k, v]) => {
      const line = k.replace("p_over_", "").replace("_total", "");
      return {
        threshold: `O ${line}`,
        probability: (typeof v === "number" ? v : 0) * 100,
      };
    })
    .slice(0, 4);

  const sotData = Object.entries(sot)
    .filter(([k]) => k.startsWith("p_over_") && k.endsWith("_total"))
    .map(([k, v]) => {
      const line = k.replace("p_over_", "").replace("_total", "");
      return {
        threshold: `O ${line}`,
        probability: (typeof v === "number" ? v : 0) * 100,
      };
    })
    .slice(0, 4);

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">Extra Statistics</div>
      </div>
      <div className="p-6 pt-2">
        <div className="grid gap-3 mb-6 md:grid-cols-2 lg:grid-cols-4">
          <StatOverview
            label="Corners"
            expected={typeof corners.expected === "string" ? corners.expected : "—"}
            median={typeof corners.total_median === "number" ? corners.total_median : null}
            iqr={typeof corners.total_q25_q75 === "string" ? corners.total_q25_q75 : null}
            icon={CornerDownRight}
          />
          <StatOverview
            label="Cards"
            expected={typeof cards.expected === "string" ? cards.expected : "—"}
            median={typeof cards.total_median === "number" ? cards.total_median : null}
            iqr={typeof cards.total_q25_q75 === "string" ? cards.total_q25_q75 : null}
            icon={CreditCard}
          />
          <StatOverview
            label="Shots"
            expected={typeof shots.expected === "string" ? shots.expected : "—"}
            median={typeof shots.total_median === "number" ? shots.total_median : null}
            iqr={typeof shots.total_q25_q75 === "string" ? shots.total_q25_q75 : null}
            icon={Target}
          />
          <StatOverview
            label="Shots on Target"
            expected={typeof sot.expected === "string" ? sot.expected : "—"}
            median={typeof sot.total_median === "number" ? sot.total_median : null}
            iqr={typeof sot.total_q25_q75 === "string" ? sot.total_q25_q75 : null}
            icon={Crosshair}
          />
        </div>

        <Tabs.Root defaultValue="corners" className="w-full">
          <Tabs.List className="grid w-full grid-cols-4 bg-secondary rounded-lg p-1">
            <Tabs.Trigger
              value="corners"
              className="px-3 py-1.5 text-sm font-medium rounded-md data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=inactive]:text-muted-foreground transition-colors"
            >
              Corners
            </Tabs.Trigger>
            <Tabs.Trigger
              value="cards"
              className="px-3 py-1.5 text-sm font-medium rounded-md data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=inactive]:text-muted-foreground transition-colors"
            >
              Cards
            </Tabs.Trigger>
            <Tabs.Trigger
              value="shots"
              className="px-3 py-1.5 text-sm font-medium rounded-md data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=inactive]:text-muted-foreground transition-colors"
            >
              Shots
            </Tabs.Trigger>
            <Tabs.Trigger
              value="sot"
              className="px-3 py-1.5 text-sm font-medium rounded-md data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=inactive]:text-muted-foreground transition-colors"
            >
              SOT
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="corners" className="mt-4">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={cornersData}>
                <XAxis dataKey="threshold" stroke={CHART_COLORS.text} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  stroke={CHART_COLORS.text}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  formatter={(v: number) => [`${v.toFixed(1)}%`]}
                />
                <Bar dataKey="probability" fill={CHART_COLORS.chart1} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Tabs.Content>

          <Tabs.Content value="cards" className="mt-4">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={cardsData}>
                <XAxis dataKey="threshold" stroke={CHART_COLORS.text} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  stroke={CHART_COLORS.text}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  formatter={(v: number) => [`${v.toFixed(1)}%`]}
                />
                <Bar dataKey="probability" fill={CHART_COLORS.chart3} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Tabs.Content>

          <Tabs.Content value="shots" className="mt-4">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={shotsData}>
                <XAxis dataKey="threshold" stroke={CHART_COLORS.text} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  stroke={CHART_COLORS.text}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  formatter={(v: number) => [`${v.toFixed(1)}%`]}
                />
                <Bar dataKey="probability" fill={CHART_COLORS.chart2} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Tabs.Content>

          <Tabs.Content value="sot" className="mt-4">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sotData}>
                <XAxis dataKey="threshold" stroke={CHART_COLORS.text} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  stroke={CHART_COLORS.text}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                  formatter={(v: number) => [`${v.toFixed(1)}%`]}
                />
                <Bar dataKey="probability" fill={CHART_COLORS.chart4} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}
