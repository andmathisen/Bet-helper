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

// Better display names for features
const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  // xG features
  "away_xg_against": "Away Team's xG Conceded",
  "away_xg_for": "Away Team's xG Scored",
  "home_xg_against": "Home Team's xG Conceded",
  "home_xg_for": "Home Team's xG Scored",
  
  // Corners
  "away_corners_against": "Away Team's Corners Conceded",
  "away_corners_for": "Away Team's Corners Won",
  "home_corners_against": "Home Team's Corners Conceded",
  "home_corners_for": "Home Team's Corners Won",
  
  // Shots
  "away_shots_against": "Away Team's Shots Conceded",
  "away_shots_for": "Away Team's Shots Taken",
  "home_shots_against": "Home Team's Shots Conceded",
  "home_shots_for": "Home Team's Shots Taken",
  
  // Shots on Target
  "away_sot_against": "Away Team's Shots on Target Conceded",
  "away_sot_for": "Away Team's Shots on Target",
  "home_sot_against": "Home Team's Shots on Target Conceded",
  "home_sot_for": "Home Team's Shots on Target",
  
  // Cards
  "away_cards_against": "Away Team's Cards Received",
  "away_cards_for": "Away Team's Cards Caused",
  "home_cards_against": "Home Team's Cards Received",
  "home_cards_for": "Home Team's Cards Caused",
  
  // Form - Goals Scored
  "away_avg_scored_away_5": "Away: Goals Scored (away, last 5)",
  "away_avg_scored_away_10": "Away: Goals Scored (away, last 10)",
  "away_avg_scored_home_5": "Away: Goals Scored (home, last 5)",
  "away_avg_scored_home_10": "Away: Goals Scored (home, last 10)",
  "home_avg_scored_home_5": "Home: Goals Scored (home, last 5)",
  "home_avg_scored_home_10": "Home: Goals Scored (home, last 10)",
  "home_avg_scored_away_5": "Home: Goals Scored (away, last 5)",
  "home_avg_scored_away_10": "Home: Goals Scored (away, last 10)",
  
  // Form - Goals Conceded
  "away_avg_conceded_away_5": "Away: Goals Conceded (away, last 5)",
  "away_avg_conceded_away_10": "Away: Goals Conceded (away, last 10)",
  "away_avg_conceded_home_5": "Away: Goals Conceded (home, last 5)",
  "away_avg_conceded_home_10": "Away: Goals Conceded (home, last 10)",
  "home_avg_conceded_home_5": "Home: Goals Conceded (home, last 5)",
  "home_avg_conceded_home_10": "Home: Goals Conceded (home, last 10)",
  "home_avg_conceded_away_5": "Home: Goals Conceded (away, last 5)",
  "home_avg_conceded_away_10": "Home: Goals Conceded (away, last 10)",
  
  // Points per game
  "away_ppg_5": "Away: Points per Game (last 5)",
  "away_ppg_10": "Away: Points per Game (last 10)",
  "home_ppg_5": "Home: Points per Game (last 5)",
  "home_ppg_10": "Home: Points per Game (last 10)",
  
  // Days since last match
  "away_days_since_last_match": "Away: Days Since Last Match",
  "home_days_since_last_match": "Home: Days Since Last Match",
  
  // Home advantage
  "home_advantage": "Home Advantage",
};

// Contextual explanations for features
const getFeatureExplanation = (
  featureName: string,
  contribution: number,
  outcome: "home" | "draw" | "away"
): string => {
  const direction = contribution > 0 ? "increases" : "decreases";
  const absContribution = Math.abs(contribution).toFixed(2);
  const outcomeLabel = outcome === "home" ? "home" : outcome === "away" ? "away" : "draw";

  // xG features
  if (featureName === "away_xg_against") {
    return `When the away team concedes more xG (weaker defense), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "away_xg_for") {
    return `When the away team scores more xG (stronger attack), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "home_xg_against") {
    return `When the home team concedes more xG (weaker defense), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "home_xg_for") {
    return `When the home team scores more xG (stronger attack), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Corners
  if (featureName === "away_corners_against") {
    return `When the away team concedes more corners, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "away_corners_for") {
    return `When the away team wins more corners, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "home_corners_against") {
    return `When the home team concedes more corners, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "home_corners_for") {
    return `When the home team wins more corners, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Shots
  if (featureName.startsWith("away_shots_")) {
    const type = featureName.endsWith("_for") ? "takes more shots" : "concedes more shots";
    return `When the away team ${type}, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName.startsWith("home_shots_")) {
    const type = featureName.endsWith("_for") ? "takes more shots" : "concedes more shots";
    return `When the home team ${type}, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Shots on Target
  if (featureName.startsWith("away_sot_")) {
    const type = featureName.endsWith("_for") ? "has more shots on target" : "concedes more shots on target";
    return `When the away team ${type}, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName.startsWith("home_sot_")) {
    const type = featureName.endsWith("_for") ? "has more shots on target" : "concedes more shots on target";
    return `When the home team ${type}, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Goals scored/conceded in form
  if (featureName.includes("avg_scored")) {
    const team = featureName.startsWith("away") ? "away" : "home";
    const venue = featureName.includes("_away_") ? "away" : "home";
    const period = featureName.includes("_10") ? "last 10 matches" : "last 5 matches";
    return `When the ${team} team has scored more goals in ${venue} matches (${period}), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName.includes("avg_conceded")) {
    const team = featureName.startsWith("away") ? "away" : "home";
    const venue = featureName.includes("_away_") ? "away" : "home";
    const period = featureName.includes("_10") ? "last 10 matches" : "last 5 matches";
    return `When the ${team} team has conceded fewer goals in ${venue} matches (${period}), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Points per game
  if (featureName.includes("ppg")) {
    const team = featureName.startsWith("away") ? "away" : "home";
    const period = featureName.includes("_10") ? "last 10 matches" : "last 5 matches";
    return `When the ${team} team has more points per game (${period}), this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Days since last match
  if (featureName === "away_days_since_last_match") {
    return `When the away team has more/less days rest since their last match, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }
  if (featureName === "home_days_since_last_match") {
    return `When the home team has more/less days rest since their last match, this ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Home advantage
  if (featureName === "home_advantage") {
    return `The home team advantage ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
  }

  // Generic fallback
  const displayName = FEATURE_DISPLAY_NAMES[featureName] || featureName.replace(/_/g, " ");
  return `${displayName} ${direction} the ${outcomeLabel} win probability by ${absContribution}%.`;
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
    // Use better display name if available
    if (FEATURE_DISPLAY_NAMES[name]) {
      return FEATURE_DISPLAY_NAMES[name];
    }
    
    // Fallback to formatted original name
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
      feature: f.feature, // Keep original name for lookups
      displayName: formatFeatureName(f.feature),
      contribution: f.contribution * 100, // Convert to percentage
      direction: f.direction,
      explanation: getFeatureExplanation(f.feature, f.contribution * 100, outcome),
    }));

    // Separate positive and negative
    const positiveFeatures = data.filter((d) => d.contribution > 0);
    const negativeFeatures = data.filter((d) => d.contribution <= 0);

    const diffFromBaseline = (prob - baseline) * 100;
    const diffSign = diffFromBaseline >= 0 ? "+" : "";
    const baselinePercent = (baseline * 100).toFixed(1);

    return (
      <div className="mb-6 last:mb-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-md font-semibold text-foreground">{label} Win</h3>
            <div className="flex items-center gap-1">
              <span className="text-sm text-muted-foreground">
                ({diffSign}{diffFromBaseline.toFixed(1)}% from baseline)
              </span>
              <div className="group/baseline relative inline-block">
                <Info className="h-3.5 w-3.5 text-muted-foreground opacity-60 hover:opacity-100 cursor-help transition-opacity" />
                <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 pointer-events-none group-hover/baseline:opacity-100 transition-opacity whitespace-nowrap z-50 w-max max-w-xs">
                  Baseline: {baselinePercent}% - This is the model's average prediction across all training data. The features above explain how this specific match differs from that average.
                  <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                </div>
              </div>
            </div>
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
                  <div key={idx} className="flex items-center justify-between text-sm group relative">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-foreground truncate">{f.displayName}</span>
                      <div className="group/icon relative inline-block flex-shrink-0">
                        <Info className="h-3.5 w-3.5 text-muted-foreground opacity-60 hover:opacity-100 cursor-help transition-opacity" />
                        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 pointer-events-none group-hover/icon:opacity-100 transition-opacity whitespace-normal z-50 w-64">
                          {f.explanation}
                          <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                        </div>
                      </div>
                    </div>
                    <span className="text-emerald-400 font-medium flex-shrink-0 ml-2">+{f.contribution.toFixed(2)}%</span>
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
                  <div key={idx} className="flex items-center justify-between text-sm group relative">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-foreground truncate">{f.displayName}</span>
                      <div className="group/icon relative inline-block flex-shrink-0">
                        <Info className="h-3.5 w-3.5 text-muted-foreground opacity-60 hover:opacity-100 cursor-help transition-opacity" />
                        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 pointer-events-none group-hover/icon:opacity-100 transition-opacity whitespace-normal z-50 w-64">
                          {f.explanation}
                          <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                        </div>
                      </div>
                    </div>
                    <span className="text-red-400 font-medium flex-shrink-0 ml-2">{f.contribution.toFixed(2)}%</span>
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
          <div className="group/header relative inline-block">
            <Info className="h-4 w-4 text-muted-foreground opacity-60 hover:opacity-100 cursor-help transition-opacity" />
            <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 pointer-events-none group-hover/header:opacity-100 transition-opacity whitespace-normal z-50 w-64">
              SHAP values show how each feature contributes to the prediction probability
              <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
            </div>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Feature contributions that push probability up or down from the baseline (the model's average prediction). Hover over features and the baseline label for more details.
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