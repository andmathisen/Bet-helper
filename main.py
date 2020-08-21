from scrape import scrape
from match import Match
from match import Team
import re

teams = {}
matchesList = []
def stringToTeam(team):
    pos = team.split('.',1)
    rest = pos[1].split(' ')
    pos = pos[0]
    rest = [string for string in rest if string != ""]

    if(len(rest) == 8):
        name = rest[0] + rest[1]
        wins = rest[3]
        draws = rest[4]
        losses = rest[5]
        goals = rest[6].split(":")
        scored = goals[0]
        conceded = goals[1]
        points = rest[7]

        team = Team(pos,wins,draws,losses,name,scored,conceded,points)

    else:
        name = rest[0]
        wins = rest[2]
        draws = rest[3]
        losses = rest[4]
        goals = rest[5].split(":")
        scored = goals[0]
        conceded = goals[1]
        points = rest[6]

        team = Team(pos,wins,draws,losses,name,scored,conceded,points)

    return team

def stringToMatch(date,matches):
    matchesRet = []
    for match in matches:

        match = match.split(" ")
        r = re.compile("\d:\d")
        match = [i for i in match if i]
        if any(r.match(c) for c in match):

            if(len(match) == 10):
                homeTeam = match[1] + match[2]
                awayTeam = match[4] + match[5]
                homeTeam = teams[homeTeam]
                awayTeam = teams[awayTeam]
                score = match[6].split(":")
                hGoals = 0
                bGoals = 0
                if(len(score)==2):
                    hGoals = score[0]
                    bGoals = score[1]
                else:
                    break
                hOdds = match[7]
                uOdds = match[8]
                bOdds = match[9]

            elif(len(match) == 9):
                if(match[3] == '-'):
                    homeTeam = match[1] + match[2]
                    awayTeam = match[4]
                    homeTeam = teams[homeTeam]
                    awayTeam = teams[awayTeam]
                    score = match[5].split(":")
                    hGoals = 0
                    bGoals = 0
                    if(len(score)==2):
                        hGoals = score[0]
                        bGoals = score[1]
                    else:
                        break
                    hOdds = match[6]
                    uOdds = match[7]
                    bOdds = match[8]
                else:
                    homeTeam = match[1]
                    awayTeam = match[3] + match[4]
                    homeTeam = teams[homeTeam]
                    awayTeam = teams[awayTeam]
                    score = match[5].split(":")
                    hGoals = 0
                    bGoals = 0
                    if(len(score)==2):
                        hGoals = score[0]
                        bGoals = score[1]
                    else:
                        break
                    hOdds = match[6]
                    uOdds = match[7]
                    bOdds = match[8]
            else:
                homeTeam = match[1]
                awayTeam = match[3]
                homeTeam = teams[homeTeam]
                awayTeam = teams[awayTeam]
                score = match[4].split(":")
                hGoals = 0
                bGoals = 0
                if(len(score)==2):
                    hGoals = score[0]
                    bGoals = score[1]
                else:
                    break
                hOdds = match[5]
                uOdds = match[6]
                bOdds = match[7]

            m = Match(date,homeTeam,awayTeam,hGoals,bGoals,hOdds,uOdds,bOdds)
            homeTeam.updateForm(m)
            awayTeam.updateForm(m)
            matchesRet.append(m)

    return matchesRet

def predictMatch(home, away):
    h = 0.33
    u = 0.33
    b = 0.33
    hwrAll5,hwrHome5,hwrAway5,hscHome5,hscAway5,hcoHome5,hcoAway5 = home.getNForm(4)
    hwrAll10,hwrHome10,hwrAway10,hscHome10,hscAway10,hcoHome10,hcoAway10 = home.getNForm(4)

    awrAll5,awrHome5,awrAway5,ascHome5,ascAway5,acoHome5,acoAway5 = away.getNForm(4)
    awrAll10,awrHome10,awrAway10,ascHome10,ascAway10,acoHome10,acoAway10 = away.getNForm(4)

    muWRAll = 0
    muWRHometoAway = 0
    scored_home_plus_conceded_away = 0
    scored_away_conceded_home = 0
    scored_home_plus_conceded_away10 = 0
    scored_away_conceded_home10 = 0

    plusPercent5 = 0.075
    minusPercent5 = 0.075/2
    plusPercent10 = 0.0375
    minusPercent10 = 0.0375/2
    difference = 0.55
    muWRAll = hwrAll5 - awrAll5
    print("match up Wr all", muWRAll)
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
    print("match up Wr home away", muWRHometoAway)
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

    score5 = scored_home_plus_conceded_away-scored_away_conceded_home
    print("scored_home_plus_conceded_away-scored_away_conceded_home", score5)
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

    ###############
    muWRAll = hwrAll10 - awrAll10
    print("match up Wr all 10 matches", muWRAll)
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
    print("match up Wr home away 10 matches", muWRHometoAway)
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

    scored_home_plus_conceded_away10 = (hscHome10 + acoAway10)/2
    scored_away_conceded_home10 = (ascAway10 + hcoHome10)/2

    score10 = scored_home_plus_conceded_away10-scored_away_conceded_home10
    print("scored_home_plus_conceded_away-scored_away_conceded_home 10 matches", score10 )
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

    print("h:", h)
    print("u:",u)
    print("b:",b)

    print("Expected result last 5 m:", str(scored_home_plus_conceded_away), "-",str(scored_away_conceded_home))
    print("Expected result last 10 m:", str(scored_home_plus_conceded_away10), "-",str(scored_away_conceded_home10))

def main(teams , matchesList):
    standings, results = scrape()

    standings = standings[1:]
    for team in standings:
        teamClass = stringToTeam(team)
        teams[teamClass.getName()] = teamClass



    for matches in results:
        matchClasses = stringToMatch(matches[0],matches[1:])
        matchesList+=matchClasses


    home = teams['Torino']

    away = teams['Verona']

    print(home.getName(),"-",away.getName())
    predictMatch(home,away)
    """
    winrateAll,winRateHome,winRateAway,scoredHome,concededHome,scoredAway,concededAway = ars.getNForm(5)
    ars.printTeam()
    print("Win rate all:",winrateAll)
    print("Win rate home:",winRateHome)
    print("Win rate away:",winRateAway)
    print("Scored home:",scoredHome)
    print("Conceded home:",concededHome)
    print("Scored away:",scoredAway)
    print("Conceded away",concededAway)
    """
main(teams, matchesList)
