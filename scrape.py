from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from match import FutureMatch
import numpy as np
import re
import time


def getResults(xpath, driver):
    list = []
    sublist = []
    counter = 0
    pattern = re.compile(".*1\sX\s2.*")

    all = driver.find_element_by_xpath(xpath)
    for line in all.find_elements_by_xpath(".//tr"):
        lineString = line.text
        lineString = lineString.replace('\n', ' ')
        lineString = lineString[:-3]
        if(pattern.match(lineString)):
            if sublist:
                list.append(sublist)
            sublist = []
            sublist.append(lineString[:-7])
        else:
            if(lineString != ''):
                sublist.append(lineString)

    return list


def getStandings(xpath, driver):
    list = []
    all = driver.find_element_by_xpath(xpath)
    for line in all.find_elements_by_xpath(".//tr"):
        lineString = line.text
        lineString = lineString.replace('\n', ' ')
        lineString = lineString[:-7]
        list.append(lineString)

    return list

def scrapeTodaysMatches(league):
    driver = webdriver.Chrome(
        executable_path="/Users/andreas/desktop/odds/chromedriver")
    regex1 = "(\d+:\d+)\s+\w+\s*(\d|\w)*\s\-\s\w+"
    regex2 = "(\d+\')\s+\w+\s\-\s\w+"
    regex3 = "(HT)\s+\w+\s\-\s\w+"
    xPath = "//*[@id='tournamentTable']"
    url = ""

    if league == "PL":
        url = "https://www.oddsportal.com/soccer/england/premier-league/"
    elif league == "SerieA":
        url= "https://www.oddsportal.com/soccer/italy/serie-a/"
    elif league == "LaLiga":
        url = "https://www.oddsportal.com/soccer/spain/laliga/"
    elif league == "Bundesliga":
        url = "https://www.oddsportal.com/soccer/germany/bundesliga/"
    elif league == "Eliteserien":
        url = "https://www.oddsportal.com/soccer/norway/eliteserien/"
    elif league == "Obos":
        url = "https://www.oddsportal.com/soccer/norway/obos-ligaen/"
    elif league == "Championship":
        url = "https://www.oddsportal.com/soccer/england/championship/"
    elif league == "LigaProfesional":
        url = "https://www.oddsportal.com/soccer/argentina/liga-profesional/"
    elif league == "Allsvenskan":
        url = "https://www.oddsportal.com/soccer/sweden/allsvenskan/"
    elif league == "Iceland":
        url = "https://www.oddsportal.com/soccer/iceland/pepsideild/"
    elif league == "LigaPortugal":
        url = "https://www.oddsportal.com/soccer/portugal/liga-portugal/"
    elif league == "Ligue2":
        url = "https://www.oddsportal.com/soccer/france/ligue-2/"
    elif league == "LeagueOne":
        url = "https://www.oddsportal.com/soccer/england/league-one/"

    driver.get(url)
    matchesList= []
    all = driver.find_element_by_xpath(xPath)
    all = all.find_elements_by_xpath(".//tr")[3:]
    for line in all:
        lineString = line.text
        lineString = lineString.replace('\n', ' ')

        if (re.match("\w+\s,\s\d+\s\w+( 1 X 2 B's)",lineString) == None) and (re.match("\d+\s\w{3}\s\w{4}( 1 X 2 B's)",lineString) == None):
            matchString = lineString.split()
            if (re.match("((\d+:\d+)|(\d+\'))\s(\D|\s|\.)+\-{1}(\s|\D)+(\d:\d)((\s\d+\.\d+){3}\s\d+)",lineString)):
                matchString.pop(-5)
            matchString = matchString[1:]

            hOdds = matchString[-4]
            uOdds = matchString[-3]
            bOdds = matchString[-2]

            matchString = matchString[:-4]
            i = 0
            while matchString[i] != '-':
                i = i+1

            home = ''.join(matchString[0:i])

            away = ''.join(matchString[i+1:])
            matchObj = FutureMatch(home,away, hOdds,uOdds,bOdds)
            matchesList.append(matchObj)
        else:
            print(lineString)
            break



    return matchesList
def scrape(league):
    driver = webdriver.Chrome(
        executable_path="/Users/andreas/desktop/odds/chromedriver")

    xPathResults = "//*[@id='tournamentTable']"
    xPathStandings = "//*[@id='table-type-1']"
    urlResults = ""
    urlStandings = ""

    if league == "PL":
        urlResults = "https://www.oddsportal.com/soccer/england/premier-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/england/premier-league/standings/"
    elif league == "SerieA":
        urlResults= "https://www.oddsportal.com/soccer/italy/serie-a/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/italy/serie-a/standings/"
    elif league == "LaLiga":
        urlResults = "https://www.oddsportal.com/soccer/spain/laliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/spain/laliga/standings/"
    elif league == "Bundesliga":
        urlResults = "https://www.oddsportal.com/soccer/germany/bundesliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/germany/bundesliga/standings/"
    elif league == "Eliteserien":
        urlResults = "https://www.oddsportal.com/soccer/norway/eliteserien/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/norway/eliteserien/standings/"
    elif league == "Obos":
        urlResults = "https://www.oddsportal.com/soccer/norway/obos-ligaen/results/#/page/standings/"
        urlStandings = "https://www.oddsportal.com/soccer/norway/obos-ligaen/standings/"
    elif league == "Championship":
        urlResults = "https://www.oddsportal.com/soccer/england/championship/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/england/championship/standings/"
    elif league == "LigaProfesional":
        urlResults = "https://www.oddsportal.com/soccer/argentina/liga-profesional/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/argentina/liga-profesional/standings/"
    elif league == "Allsvenskan":
        urlResults = "https://www.oddsportal.com/soccer/sweden/allsvenskan/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/sweden/allsvenskan/standings/"
    elif league == "Iceland":
        urlResults = "https://www.oddsportal.com/soccer/iceland/pepsideild/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/iceland/pepsideild/standings/"
    elif league == "LigaPortugal":
        urlResults = "https://www.oddsportal.com/soccer/portugal/liga-portugal/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/portugal/liga-portugal/standings/"
    elif league == "Ligue2":
        urlResults = "https://www.oddsportal.com/soccer/france/ligue-2/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/france/ligue-2/standings/"
    elif league == "LeagueOne":
        urlResults = "https://www.oddsportal.com/soccer/england/league-one/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/england/league-one/standings/"
    listResults = []
    listStandings = []

    for i in range(1, 4):
        driver.get(urlResults+str(i))
        time.sleep(5)
        listResults += getResults(xPathResults, driver)

    driver.get(urlStandings)
    time.sleep(5)
    listStandings = getStandings(xPathStandings, driver)

    return listStandings, listResults
