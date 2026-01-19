import { cn } from "@/lib/utils";
import type { Prediction } from "@/ui/types";
import { TrendingUp, TrendingDown } from "lucide-react";
import { maxEv } from "@/ui/utils";

interface MatchSidebarProps {
  matches: Prediction[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

export function MatchSidebar({ matches, selectedIndex, onSelect }: MatchSidebarProps) {
  return (
    <aside className="sticky top-0 h-screen w-72 shrink-0 border-r border-border/50 bg-card/30 overflow-y-auto">
      <div className="flex h-14 items-center border-b border-border/50 px-4 sticky top-0 bg-card/50 backdrop-blur z-10">
        <h2 className="text-sm font-semibold text-foreground">Matches</h2>
        <span className="ml-2 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
          {matches.length}
        </span>
      </div>
      <div className="space-y-1 p-2">
        {matches.map((match, idx) => {
          const best = maxEv(match);
          const bestEV = best?.ev ?? -999;
          const isPositiveEV = bestEV > 0;

          return (
            <button
              key={idx}
              onClick={() => onSelect(idx)}
              className={cn(
                "w-full rounded-lg p-3 text-left transition-all",
                selectedIndex === idx ? "bg-primary/20 ring-1 ring-primary/50" : "hover:bg-secondary/50",
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <p
                    className={cn(
                      "truncate text-sm font-medium",
                      selectedIndex === idx ? "text-foreground" : "text-foreground/80",
                    )}
                  >
                    {match.match.home}
                  </p>
                  <p
                    className={cn(
                      "truncate text-sm",
                      selectedIndex === idx ? "text-muted-foreground" : "text-muted-foreground/70",
                    )}
                  >
                    vs {match.match.away}
                  </p>
                </div>
                <div
                  className={cn(
                    "flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium",
                    isPositiveEV ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400",
                  )}
                >
                  {isPositiveEV ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                  {bestEV > 0 ? "+" : ""}
                  {(bestEV * 100).toFixed(1)}%
                </div>
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                <span className="rounded bg-secondary/50 px-1.5 py-0.5">{match.league}</span>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
