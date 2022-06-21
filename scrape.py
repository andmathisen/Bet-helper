from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from match import FutureMatch
import numpy as np
import re
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


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
    list.append(sublist)
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
    options = webdriver.ChromeOptions()

    options.add_argument('--headless')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    regex1 = "(\d+:\d+)\s+\w+\s*(\d|\w)*\s\-\s\w+"
    regex2 = "(\d+\')\s+\w+\s\-\s\w+"
    regex3 = "(HT)\s+\w+\s\-\s\w+"
    xPath = "//*[@id='tournamentTable']"
    url = ""

    if league == "PL":
        url = "https://www.oddsportal.com/soccer/england/premier-league/"
    elif league == "SerieA":
        url= "https://www.oddsportal.com/soccer/italy/serie-a/"
    elif league == "SerieB":
        url= "https://www.oddsportal.com/soccer/italy/serie-b/"
    elif league == "LaLiga":
        url = "https://www.oddsportal.com/soccer/spain/laliga/"
    elif league == "Bundesliga":
        url = "https://www.oddsportal.com/soccer/germany/bundesliga/"
    elif league == "Bundesliga2":
        url = "https://www.oddsportal.com/soccer/germany/2-bundesliga/"
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
    elif league == "LigaPortugal2":
        url = "https://www.oddsportal.com/soccer/portugal/liga-portugal-2/"
    elif league == "Ligue1":
        url = "https://www.oddsportal.com/soccer/france/ligue-1/"
    elif league == "Ligue2":
        url = "https://www.oddsportal.com/soccer/france/ligue-2/"
    elif league == "LeagueOne":
        url = "https://www.oddsportal.com/soccer/england/league-one/"
    elif league == "Eredivisie":
        url = "https://www.oddsportal.com/soccer/netherlands/eredivisie/"
    elif league == "MLS":
        url = "https://www.oddsportal.com/soccer/usa/mls/"
    elif league == "PrimeraB":
        url = "https://www.oddsportal.com/soccer/argentina/primera-b"
    elif league == "J1-league":
        url = "https://www.oddsportal.com/soccer/japan/j1-league/"
    elif league == "J2-league":
        url = "https://www.oddsportal.com/soccer/japan/j2-league/"
    elif league == "J3-league":
        url = "https://www.oddsportal.com/soccer/japan/j3-league/"
    elif league == "K1-league":
        url = "https://www.oddsportal.com/soccer/south-korea/k-league-1/"
    elif league == "K2-league":
        url = "https://www.oddsportal.com/soccer/south-korea/k-league-2/"
    elif league == "K3-league":
        url = "https://www.oddsportal.com/soccer/south-korea/k3-league/"
    elif league == "3-liga":
        url = "https://www.oddsportal.com/soccer/germany/3-liga/"
    elif league == "PremierLeagueGhana":
        url = "https://www.oddsportal.com/soccer/ghana/premier-league" 
    elif league == "GreeceSuperLeague2":
        url = "https://www.oddsportal.com/soccer/greece/super-league-2/" 
    elif league == "BankLiga":
        url = "https://www.oddsportal.com/soccer/hungary/merkantil-bank-liga/"
    elif league == "PrimeraNacional":
        url = "https://www.oddsportal.com/soccer/argentina/primera-nacional/"
    elif league == "ICLigue1":
        url = "https://www.oddsportal.com/soccer/ivory-coast/ligue-1/"
    elif league == "SuperLeague":
        url = "https://www.oddsportal.com/soccer/switzerland/super-league/"
    elif league == "AustriaBundesliga":
        url = "https://www.oddsportal.com/soccer/austria/bundesliga/"
    elif league == "ZambiaSuperLeague":
        url = "https://www.oddsportal.com/soccer/zambia/super-league/"
    elif league == "SuperLig":
        url = "https://www.oddsportal.com/soccer/turkey/super-lig/"
    elif league == "1.Lig":
        url = "https://www.oddsportal.com/soccer/turkey/1-lig/"
    elif league == "BosniaPL":
        url = "https://www.oddsportal.com/soccer/bosnia-and-herzegovina/premier-league/"
    elif league == "JamaicaPL":
        url = "https://www.oddsportal.com/soccer/jamaica/premier-league/"
    elif league == "PrimeraDivisionChile":
        url = "https://www.oddsportal.com/soccer/chile/primera-division/"
    elif league == "A-League":
        url = "https://www.oddsportal.com/soccer/australia/a-league/"
    elif league == "AlgeriaLigue1":
        url = "https://www.oddsportal.com/soccer/algeria/ligue-1/"
    
            
    
    
    driver.get(url)
    matchesList= []
    all = driver.find_element_by_xpath(xPath)
    all = all.find_elements_by_xpath(".//tr")[3:]
    
    c = 0

    for line in all:
        
        lineString = line.text
        
        lineString = lineString.replace('\n', ' ')
        matchString = lineString.split()
        
        if "B's" not in matchString and len(matchString)>0:
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
           
            if ':' in home:
                home = home[:-3]
            if ':' in away:
                away = away[:-3]
            
            matchObj = FutureMatch(home,away, hOdds,uOdds,bOdds)
            matchesList.append(matchObj)
            c += 1
            if c > 8:
                break
            



    return matchesList
def scrape(league):
    options = webdriver.ChromeOptions()

    options.add_argument('--headless')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

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
    elif league == "SerieB":
        urlResults= "https://www.oddsportal.com/soccer/italy/serie-b/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/italy/serie-b/standings/"
    elif league == "LaLiga":
        urlResults = "https://www.oddsportal.com/soccer/spain/laliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/spain/laliga/standings/"
    elif league == "Bundesliga":
        urlResults = "https://www.oddsportal.com/soccer/germany/bundesliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/germany/bundesliga/standings/"
    elif league == "Bundesliga2":
        urlResults = "https://www.oddsportal.com/soccer/germany/2-bundesliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/germany/2-bundesliga/standings/"
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
    elif league == "LigaPortugal2":
        urlResults = "https://www.oddsportal.com/soccer/portugal/liga-portugal-2/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/portugal/liga-portugal-2/standings/"
    elif league == "Ligue1":
        urlResults = "https://www.oddsportal.com/soccer/france/ligue-1/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/france/ligue-1/standings/"
    elif league == "Ligue2":
        urlResults = "https://www.oddsportal.com/soccer/france/ligue-2/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/france/ligue-2/standings/"
    elif league == "LeagueOne":
        urlResults = "https://www.oddsportal.com/soccer/england/league-one/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/england/league-one/standings/"
    elif league == "Eredivisie":
        urlResults = "https://www.oddsportal.com/soccer/netherlands/eredivisie/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/netherlands/eredivisie/standings/"
    elif league == "MLS":
        urlResults = "https://www.oddsportal.com/soccer/usa/mls/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/usa/mls/standings/"
    elif league == "PrimeraB":
        urlResults = "https://www.oddsportal.com/soccer/argentina/primera-b/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/argentina/primera-b/standings/"
    elif league == "J1-league":
        urlResults = "https://www.oddsportal.com/soccer/japan/j1-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/japan/j1-league/standings/"
    elif league == "J2-league":
        urlResults = "https://www.oddsportal.com/soccer/japan/j2-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/japan/j2-league/standings/"
    elif league == "J3-league":
        urlResults = "https://www.oddsportal.com/soccer/japan/j3-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/japan/j3-league/standings/"
    elif league == "K1-league":
        urlResults = "https://www.oddsportal.com/soccer/south-korea/k-league-1/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/south-korea/k-league-1/standings/"
    elif league == "K2-league":
        urlResults = "https://www.oddsportal.com/soccer/south-korea/k-league-2/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/south-korea/k-league-2/standings/"
    elif league == "K3-league":
        urlResults = "https://www.oddsportal.com/soccer/south-korea/k3-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/south-korea/k3-league/standings/"
    elif league == "3-liga":
        urlResults = "https://www.oddsportal.com/soccer/germany/3-liga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/germany/3-liga/standings/"
    elif league == "PremierLeagueGhana":
        urlResults = "https://www.oddsportal.com/soccer/ghana/premier-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/ghana/premier-league/standings/"
    elif league == "GreeceSuperLeague2":
        urlResults = "https://www.oddsportal.com/soccer/greece/super-league-2/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/greece/super-league-2/standings/"
    elif league == "BankLiga":
        urlResults = "https://www.oddsportal.com/soccer/hungary/merkantil-bank-liga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/hungary/merkantil-bank-liga/standings/"
    elif league == "PrimeraNacional":
        urlResults = "https://www.oddsportal.com/soccer/argentina/primera-nacional/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/argentina/primera-nacional/standings/"
    elif league == "ICLigue1":
        urlResults = "https://www.oddsportal.com/soccer/ivory-coast/ligue-1/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/ivory-coast/ligue-1/standings/"
    elif league == "SuperLeague":
        urlResults = "https://www.oddsportal.com/soccer/switzerland/super-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/switzerland/super-league/standings/"
    elif league == "AustriaBundesliga":
        urlResults = "https://www.oddsportal.com/soccer/austria/bundesliga/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/austria/bundesliga/standings/nXhNbiaL/"
    elif league == "ZambiaSuperLeague":
        urlResults = "https://www.oddsportal.com/soccer/zambia/super-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/zambia/super-league/standings/"
    elif league == "SuperLig":
        urlResults = "https://www.oddsportal.com/soccer/turkey/super-lig/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/turkey/super-lig/standings/"
    elif league == "1.Lig":
        urlResults = "https://www.oddsportal.com/soccer/turkey/1-lig/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/turkey/1-lig/standings/"
    elif league == "BosniaPL":
        urlResults = "https://www.oddsportal.com/soccer/bosnia-and-herzegovina/premier-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/bosnia-and-herzegovina/premier-league/standings/"
    elif league == "JamaicaPL":
        urlResults = "https://www.oddsportal.com/soccer/jamaica/premier-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/jamaica/premier-league/standings/"
    elif league == "PrimeraDivisionChile":
        urlResults = "https://www.oddsportal.com/soccer/chile/primera-division/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/chile/primera-division/standings/"
    elif league == "A-League":
        urlResults = "https://www.oddsportal.com/soccer/australia/a-league/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/australia/a-league/standings/"
    elif league == "AlgeriaLigue1":
        urlResults = "https://www.oddsportal.com/soccer/algeria/ligue-1/results/#/page/"
        urlStandings = "https://www.oddsportal.com/soccer/algeria/ligue-1/standings/"
           
   
    listResults = []
    listStandings = []


    for i in range(1, 5):
        driver.get(urlResults+str(i))
        time.sleep(5)
        listResults += getResults(xPathResults, driver)

      
    driver.get(urlStandings)
    time.sleep(5)
    listStandings = getStandings(xPathStandings, driver)

    return listStandings, listResults
