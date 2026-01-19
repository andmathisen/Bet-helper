import type { Prediction } from "@/ui/types";
import { Bar, BarChart, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { TrendingUp, TrendingDown, Info } from "lucide-react";

interface ShapExplanationsProps {
  match: Prediction;
}

const CHART_COLORS = {
  positive: "#34d399",
  negative: "#f87171",
  text: "#a1a1aa",
  tooltipBg: "#27272a",
  tooltipBorder: "#3f3f46",
};

export function ShapExplanations({ match }: ShapExplanationsProps) {
  const shap = match.shap_explanations;
  
  if (!shap || !match.ml_probs) {
    return null;
  }

  const getTopFeaturesForOutcome = (outcome: "home" | "draw" | "away", limit: number = 8) => {
    const features = shap.top_features[outcome] || [];
    return features.slice(0, limit);
  };

  const formatFeatureName = (name: string): string => {
    // Convert feature names to readable labels
    return name
      .replace(/_/g, " ")
      .replace(/\b(\w)/g, (char) => char.toUpperCase())
      .replace(/Ppg/g, "PPG")
      .replace(/Xg/g, "xG")
      .replace(/Sot/g, "SOT")
      .replace(/League\s+(\w+)/g, "League: $1");
  };

  const renderOutcomeExplanation = (outcome: "home" | "draw" | "away", label: string) => {
    const features = getTopFeaturesForOutcome(outcome, 8);
    const prob = match.ml_probs?.[outcome] ?? 0;
    const baseline = shap.baseline[outcome] ?? 0.33;

    if (features.length === 0) {
      return null;
    }

    const data = features.map((f) => ({
      feature: formatFeatureName(f.feature),
      contribution: f.contribution * 100, // Convert to percentage
      direction: f.direction,
    }));

    // Separate positive and negative
    const positiveFeatures = data.filter((d) => d.contribution > 0);
    const negativeFeatures = data.filter((d) => d.contribution <= 0);

    return (
      <div className="mb-6 last:mb-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-md font-semibold text-foreground">{label} Win</h3>
            <span className="text-sm text-muted-foreground">
              ({((prob - baseline) * 100).toFixed(1)}% from baseline)
            </span>
          </div>
          <span className="text-sm font-medium text-primary">{(prob * 100).toFixed(1)}%</span>
        </div>

        <div className="space-y-3">
          {positiveFeatures.length > 0 && (
            <div>
              <div className="flex items-center gap-1 mb-2 text-xs text-muted-foreground">
                <TrendingUp className="h-3 w-3 text-emerald-400" />
                <span>Increasing probability</span>
              </div>
              <div className="space-y-1">
                {positiveFeatures.map((f, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-foreground">{f.feature}</span>
                    <span className="text-emerald-400 font-medium">+{f.contribution.toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {negativeFeatures.length > 0 && (
            <div>
              <div className="flex items-center gap-1 mb-2 text-xs text-muted-foreground">
                <TrendingDown className="h-3 w-3 text-red-400" />
                <span>Decreasing probability</span>
              </div>
              <div className="space-y-1">
                {negativeFeatures.map((f, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-foreground">{f.feature}</span>
                    <span className="text-red-400 font-medium">{f.contribution.toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg">
      <div className="pb-2 px-6 pt-6">
        <div className="text-lg font-semibold flex items-center gap-2">
          ML Model Explanation (SHAP)
          <Info className="h-4 w-4 text-muted-foreground" title="SHAP values show how each feature contributes to the prediction probability" />
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Feature contributions that push probability up or down from the baseline
        </p>
      </div>
      <div className="p-6 pt-2">
        {renderOutcomeExplanation("home", "Home")}
        {renderOutcomeExplanation("draw", "Draw")}
        {renderOutcomeExplanation("away", "Away")}
      </div>
    </div>
  );
}