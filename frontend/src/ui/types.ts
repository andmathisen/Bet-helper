export type Outcome = "Home" | "Draw" | "Away" | "No bet";

export type ProbTriple = { home: number; draw: number; away: number };

export type ShapFeatureContribution = {
  feature: string;
  contribution: number;
  direction: "positive" | "negative";
};

export type ShapExplanations = {
  baseline: ProbTriple;
  home: Record<string, number>;
  draw: Record<string, number>;
  away: Record<string, number>;
  top_features: {
    home: ShapFeatureContribution[];
    draw: ShapFeatureContribution[];
    away: ShapFeatureContribution[];
  };
};

export type Prediction = {
  league: string;
  match: { home: string; away: string };
  odds: { home: number; draw: number; away: number };
  probs: ProbTriple;
  ml_probs?: ProbTriple | null;
  market_implied?: ProbTriple | null;
  blended?: ProbTriple | null;
  ev: { home: number | null; draw: number | null; away: number | null };
  ml_ev?: { home: number | null; draw: number | null; away: number | null } | null;
  recommended: { outcome: Outcome; ev: number | null; kelly_25pct: number | null };
  expected_result: string;
  distribution: Record<string, unknown>;
  extra_stats: Record<string, unknown>;
  form?: Record<string, unknown>;
  shap_explanations?: ShapExplanations | null;
};

export type PredictionsResponse = {
  league: string;
  count: number;
  predictions: Prediction[];
};

export type LeaguesResponse = {
  count: number;
  leagues: string[];
};

export type HistoricalSummaryResponse = {
  league: string;
  count: number;
  first_match_date: string | null;
  last_match_date: string | null;
  last_updated_utc: string | null;
  path: string;
};

export type ScrapeReport = {
  league: string;
  has_delta: boolean;
  new_matches_count: number;
  last_existing_date: string | null;
  upcoming_matches_count: number;
  updated_historical: boolean;
  historical_path: string;
  upcoming_path: string;
  timestamp: string;
};
