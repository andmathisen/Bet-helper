class Match:
    def __init__(self,date,homeTeam,awayTeam,hGoals,bGoals, hOdds, uOdds, bOdds):
        self.date = date
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.hGoals = hGoals
        self.bGoals = bGoals
        self.hOdds = hOdds
        self.uOdds = uOdds
        self.bOdds = bOdds

    def printMatch(self):
        print(self.date + ":\n")
        print("     home:" + self.homeTeam.getName() + "\n")
        print("     away:" + self.awayTeam.getName() + "\n")
        print("     hGoals:" + self.hGoals+ "\n")
        print("     bGoals:" + self.bGoals+ "\n")
        print("     hOdds:" + self.hOdds+ "\n")
        print("     uOdds:" + self.uOdds+ "\n")
        print("     bOdds:" + self.bOdds+ "\n")

    def getScore(self):
        return int(self.hGoals),int(self.bGoals)

    def getOdds(self):
        return float(self.hOdds),float(self.uOdds), float(self.bOdds)

    def getHomeTeam(self):
        return self.homeTeam

    def getAwayTeam(self):
        return self.awayTeam
class FutureMatch:
    def __init__(self,homeTeam,awayTeam, hOdds, uOdds, bOdds):
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.hOdds = hOdds
        self.uOdds = uOdds
        self.bOdds = bOdds

    def getOdds(self):
        return float(self.hOdds),float(self.uOdds), float(self.bOdds)

    def getHomeTeam(self):
        return self.homeTeam

    def getAwayTeam(self):
        return self.awayTeam
    def printMatch(self):
        print(self.getHomeTeam(),self.getAwayTeam(),self.getOdds())
class Team:
    def __init__(self,position,wins,draws,losses,teamName, goals,conceded,points):
        self.position = position
        self.wins = wins
        self.draws = draws
        self.losses = losses
        self.teamName = teamName
        self.goals = goals
        self.conceded = conceded
        self.points = points
        self.form = []

    def updateForm(self,match):
        self.form.append(match)

    def getNForm(self,nMatches):
        matchesHome = 0
        matchesAway = 0
        winRateAll = 0
        winRateHome = 0
        winRateAway = 0
        scoredHome = 0
        scoredAway = 0
        concededHome = 0
        concededAway = 0
        for i in range(nMatches):
            m = self.form[i]
            hGoals, bGoals = m.getScore()
            h,u,b = m.getOdds()
            hOddsBoost = 1+(h*0.1)
            uOddsBoost = 1+(u*0.1)
            bOddsBoost = 1+(b*0.1)

            if self is m.getHomeTeam():

                matchesHome += 1
                scoredHome += hGoals
                concededHome += bGoals
                if hGoals == bGoals:
                    winRateAll +=1
                    winRateHome +=1
                elif hGoals > bGoals:
                    winRateAll +=3
                    winRateHome +=3


            else:
                matchesAway +=1
                scoredAway += bGoals
                concededAway += hGoals
                if hGoals == bGoals:
                    winRateAll +=1
                    winRateAway +=1
                elif hGoals < bGoals:
                    winRateAll +=3
                    winRateAway +=3




        return round(winRateAll/(nMatches),2),round(winRateHome/matchesHome,2),round(winRateAway/matchesAway,2),round(scoredHome/matchesHome,2),round(scoredAway/matchesAway,2),round(concededHome/matchesHome,2),round(concededAway/matchesAway,2)

    def printNLastMatches(self,n):
            for i in range(n):
                m = self.form[i]
                m.printMatch()
    def getName(self):
        return self.teamName

    def printTeam(self):
        print(self.teamName + ":\n")
        print("     pos:" + self.position + "\n")
        print("     wins:" + self.wins+ "\n")
        print("     draws:" + self.draws+ "\n")
        print("     losses:" + self.losses+ "\n")
        print("     homeGoals:" + self.goals+ "\n")
        print("     conceded:" + self.conceded+ "\n")
        print("     points:" + self.points+ "\n")
