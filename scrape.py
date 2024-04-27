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

leagues = {
        "PL":"/england/premier-league/",
        "SerieA":"/italy/serie-a/",
        "SerieB":"/italy/serie-b/",
        "LaLiga":"/spain/laliga/",
        "Bundesliga":"/germany/bundesliga/",
        "Bundesliga2":"/germany/2-bundesliga/",
        "Eliteserien":"/norway/eliteserien/",
        "Obos":"/norway/obos-ligaen/",
        "Championship":"/england/championship/",
        "LigaProfesional":"/argentina/liga-profesional/",
        "Allsvenskan":"/sweden/allsvenskan/",
        "Iceland":"/iceland/pepsideild/",
        "LigaPortugal":"/portugal/liga-portugal/",
        "LigaPortugal2":"/portugal/liga-portugal-2/",
        "Ligue1":"/france/ligue-1/",
        "Ligue2":"/france/ligue-2/",
        "LeagueOne":"/england/league-one/",
        "Eredivisie":"/netherlands/eredivisie/",
        "MLS":"/usa/mls/",
        "PrimeraB":"/argentina/primera-b",
        "J1-league":"/japan/j1-league/",
        "J2-league":"/japan/j2-league/",
        "J3-league":"/japan/j3-league/",
        "K1-league":"/south-korea/k-league-1/",
        "K2-league":"/south-korea/k-league-2/",
        "K3-league":"/south-korea/k3-league/",
        "3-liga":"/germany/3-liga/",
        "PremierLeagueGhana":"/ghana/premier-league",
        "GreeceSuperLeague2":"/greece/super-league-2/", 
        "BankLiga":"/hungary/merkantil-bank-liga/",
        "PrimeraNacional":"/argentina/primera-nacional/",
        "ICLigue1":"/ivory-coast/ligue-1/",
        "SuperLeague":"/switzerland/super-league/",
        "AustriaBundesliga":"/austria/bundesliga/",
        "ZambiaSuperLeague":"/zambia/super-league/",
        "SuperLig":"/turkey/super-lig/",
        "1.Lig":"/turkey/1-lig/",
        "BosniaPL":"/bosnia-and-herzegovina/premier-league/",
        "JamaicaPL":"/jamaica/premier-league/",
        "PrimeraDivisionChile":"/chile/primera-division/",
        "A-League":"/australia/a-league/",
        "AlgeriaLigue1":"/algeria/ligue-1/"
    }

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
    url = f"https://www.oddsportal.com/soccer{leagues[league]}"
    
    
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
    urlResults = f"https://www.oddsportal.com/soccer{leagues[league]}results/#/page/"
    urlStandings = f"https://www.oddsportal.com/soccer{leagues[league]}standings/"

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
