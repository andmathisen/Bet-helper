from scrape import scrape,scrapeTodaysMatches
from match import Match
from match import Team
import re
import sys
from simple_term_menu import TerminalMenu
import json

teams = {}
matchesList = []

def RepresentsInt(s):
    if(s[0] == "0"):
        return False
    else:
        try:
            int(s)
            return True
        except ValueError:
            return False

def stringToTeam(team):
    
    pos = team.split('.', 1)
    rest = pos[1].split(' ')
    pos = pos[0]
    
    rest = [string for string in rest if string != ""]
    
    name = ""
    if len(rest) < 6:
        return None
    for subname in rest[:-6]:
        name += subname
    matchesPlayed = rest[-6]
    wins = rest[-5]
    draws = rest[-4]
    losses = rest[-3]
    goals = rest[-2].split(":")

    if len(goals)<2:
        return None
    scored = goals[0]
    conceded = goals[1]

    points = rest[-1]

    team = Team(pos, wins, draws, losses, name, scored, conceded, points)

    return team





def stringToMatch(date, matches):
    matchesRet = []
    
    for match in matches:

        match = match.split(" ")
        r = re.compile("\d:\d")
        match = [i for i in match if i]
        if any(r.match(c) for c in match):
            date = match[0]
            i = 1
            homeTeamName = ""
            awayTeamName = ""
            homeTeamObj = None
            awayTeamObj = None
            while(match[i] != '-'):
                homeTeamName += match[i]
                i += 1
            i+=1
            while(r.match(match[i]) == None):
                awayTeamName += match[i]
                i+=1
            
            if i+3 >= len(match):
                return None
            score = match[i].split(":")
            hGoals = score[0]
            aGoals = score[1]
            hOdds = match[i+1]
            uOdds = match[i+2]
            bOdds = match[i+3]

            homeTeamObj = teams[homeTeamName]
            awayTeamObj = teams[awayTeamName]
            m = Match(date, homeTeamObj, awayTeamObj, hGoals, aGoals, hOdds, uOdds, bOdds)
            homeTeamObj.updateForm(m)
            awayTeamObj.updateForm(m)
            matchesRet.append(m)


    return matchesRet




def NaivePredictMatch(home, away):
    h = 0.33
    u = 0.33
    b = 0.33

    # if len(home.form) < 5:
    #     home.printNLastMatches(4)
    # if len(away.form) < 5:
    #     away.printNLastMatches(4)

    hwrAll10, hwrHome10, hwrAway10, hscHome10, hscAway10, hcoHome10, hcoAway10 = home.getNForm(
        10)

    hwrAll5, hwrHome5, hwrAway5, hscHome5, hscAway5, hcoHome5, hcoAway5 = home.getNForm(
        5)
    
    awrAll10, awrHome10, awrAway10, ascHome10, ascAway10, acoHome10, acoAway10 = away.getNForm(
        10)
    
    awrAll5, awrHome5, awrAway5, ascHome5, ascAway5, acoHome5, acoAway5 = away.getNForm(
        5)

    hwrAll3, hwrHome3, hwrAway3, hscHome3, hscAway3, hcoHome3, hcoAway3 = home.getNForm(
        3)
    
    awrAll3, awrHome3, awrAway3, ascHome3, ascAway3, acoHome3, acoAway3 = away.getNForm(
        3)
    
    muWRAll = 0
    muWRHometoAway = 0
    scored_home_plus_conceded_away = 0
    scored_away_conceded_home = 0
    tot_scored_home_plus_conceded_away = 0
    tot_scored_away_conceded_home = 0
    plusPercent3 = 0.0425
    minusPercent3 = 0.0425/2
    plusPercent5 = 0.0875
    minusPercent5 = 0.0875/2
    plusPercent10 = 0.02125
    minusPercent10 = 0.02125/2
    difference = 0.55
    muWRAll = hwrAll3 - awrAll3
    if abs(muWRAll) < difference:
        h = h-minusPercent3
        u = u+plusPercent3
        b = b-minusPercent3
    elif muWRAll < 0:
        h = h-minusPercent3
        u = u-minusPercent3
        b = b+plusPercent3
    else:
        h = h+plusPercent3
        u = u-minusPercent3 
        b = b-minusPercent3

    muWRHometoAway = hwrHome3 - awrAway3
    if abs(muWRHometoAway) < difference:
        h = h-minusPercent3
        u = u+plusPercent3
        b = b-minusPercent3
    elif muWRHometoAway < 0:
        h = h-minusPercent3
        u = u-minusPercent3
        b = b+plusPercent3
    else:
        h = h+plusPercent3
        u = u-minusPercent3
        b = b-minusPercent3

    scored_home_plus_conceded_away = (hscHome3 + acoAway3)/2
    scored_away_conceded_home = (ascAway3 + hcoHome3)/2
    tot_scored_home_plus_conceded_away += scored_home_plus_conceded_away * 0.2
    tot_scored_away_conceded_home += scored_away_conceded_home * 0.2
    score3 = scored_home_plus_conceded_away-scored_away_conceded_home
    if abs(score3) < difference:
        h = h-minusPercent3
        u = u+plusPercent3
        b = b-minusPercent3
    elif score3 < 0:
        h = h-minusPercent3
        u = u-minusPercent3
        b = b+plusPercent3
    else:
        h = h+plusPercent3
        u = u-minusPercent3
        b = b-minusPercent3

    muWRAll = hwrAll5 - awrAll5
    if abs(muWRAll) < difference:
        h = h-minusPercent5
        u = u+plusPercent5
        b = b-minusPercent5
    elif muWRAll < 0:
        h = h-minusPercent5
        u = u-minusPercent5
        b = b+plusPercent5
    else:
        h = h+plusPercent5
        u = u-minusPercent5
        b = b-minusPercent5

    muWRHometoAway = hwrHome5 - awrAway5
    if abs(muWRHometoAway) < difference:
        h = h-minusPercent5
        u = u+plusPercent5
        b = b-minusPercent5
    elif muWRHometoAway < 0:
        h = h-minusPercent5
        u = u-minusPercent5
        b = b+plusPercent5
    else:
        h = h+plusPercent5
        u = u-minusPercent5
        b = b-minusPercent5

    scored_home_plus_conceded_away = (hscHome5 + acoAway5)/2
    scored_away_conceded_home = (ascAway5 + hcoHome5)/2
    tot_scored_home_plus_conceded_away += scored_home_plus_conceded_away * 0.7
    tot_scored_away_conceded_home += scored_away_conceded_home * 0.7

    score5 = scored_home_plus_conceded_away-scored_away_conceded_home
    if abs(score5) < difference:
        h = h-minusPercent5
        u = u+plusPercent5
        b = b-minusPercent5
    elif score5 < 0:
        h = h-minusPercent5
        u = u-minusPercent5
        b = b+plusPercent5
    else:
        h = h+plusPercent5
        u = u-minusPercent5
        b = b-minusPercent5

    muWRAll = hwrAll10 - awrAll10
    if abs(muWRAll) < difference:
        h = h-minusPercent10
        u = u+plusPercent10
        b = b-minusPercent10
    elif muWRAll < 0:
        h = h-minusPercent10
        u = u-minusPercent10
        b = b+plusPercent10
    else:
        h = h+plusPercent10
        u = u-minusPercent10
        b = b-minusPercent10

    muWRHometoAway = hwrHome10 - awrAway10
    if abs(muWRHometoAway) < difference:
        h = h-minusPercent10
        u = u+plusPercent10
        b = b-minusPercent10
    elif muWRHometoAway < 0:
        h = h-minusPercent10
        u = u-minusPercent10
        b = b+plusPercent10
    else:
        h = h+plusPercent10
        u = u-minusPercent10
        b = b-minusPercent10

    scored_home_plus_conceded_away = (hscHome10 + acoAway10)/2
    scored_away_conceded_home = (ascAway10 + hcoHome10)/2
    tot_scored_home_plus_conceded_away += scored_home_plus_conceded_away * 0.1
    tot_scored_away_conceded_home += scored_away_conceded_home * 0.1

    score10 = scored_home_plus_conceded_away-scored_away_conceded_home
    if abs(score10) < difference:
        h = h-minusPercent10
        u = u+plusPercent10
        b = b-minusPercent10
    elif score10 < 0:
        h = h-minusPercent10
        u = u-minusPercent10
        b = b+plusPercent10
    else:
        h = h+plusPercent10
        u = u-minusPercent10
        b = b-minusPercent10

    expectedResult = "Expected result last 5 m:" + str(
        round(tot_scored_home_plus_conceded_away,2)) + "-" + str(round(tot_scored_away_conceded_home,2))
    
    return h,u,b,expectedResult

def mainTopMatches(teams,matchesList):

    
    options = [
            "ALL",
            "PL",
            "SerieA",
            "LaLiga",
            "Bundesliga",
            "Ligue1",
            "Eliteserien",
            "Obos",
            "Championship",
            "Bundesliga2",
            "SerieB",
            "Ligue2",
            "LigaProfesional",
            "Allsvenskan",
            "Iceland",
            "LigaPortugal",
            "LigaPortugal2",
            "LeagueOne",
            "Eredivisie",
            "MLS",
            "3-liga",
            "PrimeraB",
            "J1-league",
            "J2-league",
            "J3-league",
            "K1-league",
            "K2-league",
            "K3-league",
            "PremierLeagueGhana",
            "GreeceSuperLeague2",
            "BankLiga",
            "PrimeraNacional",
            "ICLigue1",
            "SuperLeague",
            "AustriaBundesliga",
            "ZambiaSuperLeague",
            "SuperLig",
            "1.Lig",
            "BosniaPL",
            "JamaicaPL",
            "PrimeraDivisionChile",
            "A-League",
            "AlgeriaLigue1"
            ]
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()

    league = options[menu_entry_index]
    if league == "ALL":
        leagues = options[1:]
    else:
        leagues = [league]    
    all_matches = {}
    
    for league in leagues:
        valid_league = True
        standings,results = scrape(league)
        if len(standings) > 0 and len(results) > 0:
            standings = standings[1:]
            standings = [ x for x in standings if '#' not in x ]
            
            for team in standings:
                teamClass = stringToTeam(team)
                if teamClass == None:
                    valid_league = False
                    break
                teams[teamClass.getName()] = teamClass                
            if valid_league:
                for matches in results:
                    matchClasses = stringToMatch(matches[0], matches[1:])
                    if matchClasses == None:
                        valid_league == False
                        break
                    matchesList += matchClasses
                todaysMatches = scrapeTodaysMatches(league)
                for match in todaysMatches:
                    if match.getHomeTeam() in teams and match.getAwayTeam() in teams:
                        home = teams[match.getHomeTeam()]
                        away = teams[match.getAwayTeam()]

                        hPredicted,uPredicted,bPredicted,expectedResult = NaivePredictMatch(home,away)
                        hOdds,uOdds, bOdds = match.getOdds()
                        mhOdds,muOdds, mbOdds = round(hOdds*hPredicted,2),round(uOdds*uPredicted,2),round(bOdds*bPredicted,2)
                        all_matches[max([mhOdds,muOdds,mbOdds])] = {"Match":str(home.getName()+ "-" + away.getName()), "Predicted odds":str(round(hPredicted,2)) + "|" + str(round(uPredicted,2)) + "|" +str(round(bPredicted,2)),
                        "Actual odds": str(hOdds)+"|" + str(uOdds) + "|" + str(bOdds),"Multiplied odds": str(mhOdds)+"|" + str(muOdds) + "|" + str(mbOdds), "Expected result": expectedResult}
                print("Retrieved matches from: " + league)
    all_matches = dict(sorted(all_matches.items()))          
    with open("matches.json", "a") as outfile:
       
        json.dump(all_matches, outfile,indent=2)




mainTopMatches(teams, matchesList)
