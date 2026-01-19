import type { Prediction } from "@/ui/types";
import { Bar, BarChart, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface ProbabilityChartProps {
  match: Prediction;
}

const CHART_COLORS = {
  model: "#34d399",
  mlModel: "#22d3ee",
  market: "#818cf8",
  blended: "#fbbf24",
  text: "#a1a1aa",
  tooltipBg: "#27272a",
  tooltipBorder: "#3f3f46",
};

export function ProbabilityChart({ match }: ProbabilityChartProps) {
  const data = [
    {
      outcome: "Home",
      "Poisson": match.probs.home * 100,
      "ML Model": match.ml_probs?.home ? match.ml_probs.home * 100 : null,
      Market: match.market_implied?.home ? match.market_implied.home * 100 : null,
      Blended: match.blended?.home ? match.blended.home * 100 : null,
    },
    {
      outcome: "Draw",
      "Poisson": match.probs.draw * 100,
      "ML Model": match.ml_probs?.draw ? match.ml_probs.draw * 100 : null,
      Market: match.market_implied?.draw ? match.market_implied.draw * 100 : null,
      Blended: match.blended?.draw ? match.blended.draw * 100 : null,
    },
    {
      outcome: "Away",
      "Poisson": match.probs.away * 100,
      "ML Model": match.ml_probs?.away ? match.ml_probs.away * 100 : null,
      Market: match.market_implied?.away ? match.market_implied.away * 100 : null,
      Blended: match.blended?.away ? match.blended.away * 100 : null,
    },
  ].map((d) => ({
    ...d,
    "ML Model": d["ML Model"] ?? 0,
    Market: d.Market ?? 0,
    Blended: d.Blended ?? 0,
  }));

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold">Probability Comparison</div>
      </div>
      <div className="p-6 pt-2">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} barCategoryGap="20%">
            <XAxis dataKey="outcome" stroke={CHART_COLORS.text} fontSize={12} tickLine={false} axisLine={false} />
            <YAxis
              stroke={CHART_COLORS.text}
              fontSize={12}
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
              formatter={(value: number) => [`${value.toFixed(1)}%`]}
              labelStyle={{ color: "#fff" }}
            />
            <Legend wrapperStyle={{ color: "#fff" }} />
            <Bar dataKey="Poisson" fill={CHART_COLORS.model} radius={[4, 4, 0, 0]} />
            {match.ml_probs && <Bar dataKey="ML Model" fill={CHART_COLORS.mlModel} radius={[4, 4, 0, 0]} />}
            {match.market_implied && <Bar dataKey="Market" fill={CHART_COLORS.market} radius={[4, 4, 0, 0]} />}
            {match.blended && <Bar dataKey="Blended" fill={CHART_COLORS.blended} radius={[4, 4, 0, 0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
