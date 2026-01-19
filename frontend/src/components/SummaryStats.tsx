import type { Prediction } from "@/ui/types";
import { TrendingUp, AlertTriangle, CheckCircle2, BarChart3 } from "lucide-react";

interface SummaryStatsProps {
  matches: Prediction[];
}

export function SummaryStats({ matches }: SummaryStatsProps) {
  const positiveEVMatches = matches.filter((m) => {
    const ev = m.recommended.ev ?? -999;
    return ev > 0;
  });

  const avgBestEV =
    matches.reduce((sum, m) => {
      const ev = m.recommended.ev ?? -999;
      return sum + Math.max(ev, 0);
    }, 0) / matches.length;

  const avgExpectedGoals =
    matches.reduce((sum, m) => {
      const dist: any = m.distribution ?? {};
      const home = typeof dist.expected_goals_home === "number" ? dist.expected_goals_home : 0;
      const away = typeof dist.expected_goals_away === "number" ? dist.expected_goals_away : 0;
      return sum + home + away;
    }, 0) / matches.length;

  const highBTTSMatches = matches.filter((m) => {
    const dist: any = m.distribution ?? {};
    const p_btts = typeof dist.p_btts === "number" ? dist.p_btts : 0;
    return p_btts > 0.6;
  });

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <BarChart3 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Total Matches</p>
            <p className="text-2xl font-bold text-foreground">{matches.length}</p>
          </div>
        </div>
      </div>

      <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <TrendingUp className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">+EV Matches</p>
            <p className="text-2xl font-bold text-primary">{positiveEVMatches.length}</p>
          </div>
        </div>
      </div>

      <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-3/10">
            <CheckCircle2 className="h-5 w-5 text-chart-3" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Avg Best EV</p>
            <p className={`text-2xl font-bold ${avgBestEV > 0 ? "text-primary" : "text-destructive"}`}>
              {(avgBestEV * 100).toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      <div className="border border-border/50 bg-card/50 backdrop-blur rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/10">
            <AlertTriangle className="h-5 w-5 text-chart-2" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">High BTTS (60%+)</p>
            <p className="text-2xl font-bold text-foreground">{highBTTSMatches.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
