from scrape import scrape,scrapeTodaysMatches
from match import Match
from match import Team
import re
import sys

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

    i = 0
    name = ""
    while(RepresentsInt(rest[i]) != True):
        name += rest[i]
        i += 1

    matchesPlayed = rest[i]
    wins = rest[i+1]
    draws = rest[i+2]
    losses = rest[i+3]
    goals = rest[i+4].split(":")

    scored = goals[0]
    conceded = goals[1]

    points = rest[i+5]

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




def predictMatch(home, away):
    h = 0.33
    u = 0.33
    b = 0.33
    hwrAll5, hwrHome5, hwrAway5, hscHome5, hscAway5, hcoHome5, hcoAway5 = home.getNForm(
        4)
    hwrAll10, hwrHome10, hwrAway10, hscHome10, hscAway10, hcoHome10, hcoAway10 = home.getNForm(
        4)

    awrAll5, awrHome5, awrAway5, ascHome5, ascAway5, acoHome5, acoAway5 = away.getNForm(
        4)
    awrAll10, awrHome10, awrAway10, ascHome10, ascAway10, acoHome10, acoAway10 = away.getNForm(
        4)

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

    scored_home_plus_conceded_away10 = (hscHome10 + acoAway10)/2
    scored_away_conceded_home10 = (ascAway10 + hcoHome10)/2

    score10 = scored_home_plus_conceded_away10-scored_away_conceded_home10
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

    #print("h:", h)
    #print("u:", u)
    #print("b:", b)

    expectedResult = "Expected result last 5 m:" + str(
        scored_home_plus_conceded_away) + "-" + str(scored_away_conceded_home)
    #print(expectedResult)
    return h,u,b,expectedResult

def mainTopMatches(teams,matchesList):

    for i in range(1,len(sys.argv)):

        standings,results = scrape(sys.argv[i])

        standings = standings[1:]
        for team in standings:
            teamClass = stringToTeam(team)
            teams[teamClass.getName()] = teamClass

        for matches in results:
            matchClasses = stringToMatch(matches[0], matches[1:])
            matchesList += matchClasses

        todaysMatches = scrapeTodaysMatches(sys.argv[i])
        for match in todaysMatches:
            home = teams[match.getHomeTeam()]
            away = teams[match.getAwayTeam()]

            hPredicted,uPredicted,bPredicted,expectedResult = predictMatch(home,away)
            hOdds,uOdds, bOdds = match.getOdds()
            #print("Home rating:" + str(hPredicted*hOdds) + "\n" + "Tie rating:" + str(uPredicted*uOdds) + "\n" + "Away rating:" + str(bPredicted*bOdds) + "\n" + expectedResult)
            print("---------------------------------")
            print(home.getName(), "-", away.getName(),"\n")
            print("Predicted odds: ",round(hPredicted,2),"|",round(uPredicted,2),"|",round(bPredicted,2),"\n")
            print("Actual odds: ", hOdds,"|",uOdds,"|",bOdds,"\n")
            print("Multiplied odds: ",round(hOdds*hPredicted,2),"|",round(uOdds*uPredicted,2),"|",round(bOdds*bPredicted,2),"\n")
            print(expectedResult)
            print("---------------------------------")


def main(teams, matchesList):
    standings, results = scrape(sys.argv[3])


    standings = standings[1:]
    for team in standings:
        teamClass = stringToTeam(team)
        teams[teamClass.getName()] = teamClass

    for matches in results:
        matchClasses = stringToMatch(matches[0], matches[1:])
        matchesList += matchClasses

    home = teams[sys.argv[1]]
    away = teams[sys.argv[2]]

    print(home.getName(), "-", away.getName())
    predictMatch(home, away)


#main(teams, matchesList)
mainTopMatches(teams, matchesList)
