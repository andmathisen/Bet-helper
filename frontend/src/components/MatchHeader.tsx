import type { Prediction } from "@/ui/types";
import { Target, TrendingUp } from "lucide-react";

interface MatchHeaderProps {
  match: Prediction;
  evModel: "poisson" | "ml";
  onEvModelChange: (model: "poisson" | "ml") => void;
}

export function MatchHeader({ match, evModel, onEvModelChange }: MatchHeaderProps) {
  // Calculate recommended outcome and EV based on selected model
  const getBestEV = () => {
    const ev = evModel === "ml" ? match.ml_ev : match.ev;
    if (!ev) return { outcome: "No bet", ev: null };
    
    const evs = [
      { outcome: "Home" as const, ev: ev.home },
      { outcome: "Draw" as const, ev: ev.draw },
      { outcome: "Away" as const, ev: ev.away },
    ];
    
    const best = evs.reduce((best, current) => {
      if (current.ev === null) return best;
      if (best === null || current.ev > best.ev) return current;
      return best;
    }, null as { outcome: "Home" | "Draw" | "Away"; ev: number | null } | null);
    
    return best ?? { outcome: "No bet", ev: null };
  };

  const bestEV = getBestEV();
  const evColor = (bestEV.ev ?? 0) > 0 ? "text-primary" : "text-destructive";

  return (
    <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-4">
          <span className="px-2 py-1 text-xs font-medium rounded border border-primary text-primary bg-primary/10">
            {match.league}
          </span>
          <div className="text-center lg:text-left">
            <h2 className="text-xl font-bold text-foreground md:text-2xl">
              {match.match.home} <span className="text-muted-foreground">vs</span> {match.match.away}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">{match.expected_result}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 rounded-lg bg-secondary p-3">
            <Target className="h-4 w-4 text-primary" />
            <div className="text-left">
              <p className="text-xs text-muted-foreground">Recommended</p>
              <p className="font-semibold text-foreground">{bestEV.outcome}</p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-lg bg-secondary p-3">
            <TrendingUp className={`h-4 w-4 ${evColor}`} />
            <div className="text-left">
              <p className="text-xs text-muted-foreground">Expected Value ({evModel === "ml" ? "ML" : "Poisson"})</p>
              <p className={`font-semibold ${evColor}`}>
                {bestEV.ev !== null ? (bestEV.ev * 100).toFixed(2) : "—"}%
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-lg bg-secondary p-3">
            <div className="text-left">
              <p className="text-xs text-muted-foreground mb-1">EV Model</p>
              <div className="flex gap-1">
                <button
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                    evModel === "poisson"
                      ? "bg-primary text-primary-foreground"
                      : "bg-background text-muted-foreground hover:bg-secondary"
                  }`}
                  onClick={() => onEvModelChange("poisson")}
                >
                  Poisson
                </button>
                <button
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                    evModel === "ml"
                      ? "bg-primary text-primary-foreground"
                      : "bg-background text-muted-foreground hover:bg-secondary"
                  }`}
                  onClick={() => onEvModelChange("ml")}
                  disabled={!match.ml_ev}
                >
                  ML
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
