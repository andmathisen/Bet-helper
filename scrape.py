from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
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

def getStandings(xpath,driver):
    list = []
    all = driver.find_element_by_xpath(xpath)
    for line in all.find_elements_by_xpath(".//tr"):
        lineString = line.text
        lineString = lineString.replace('\n', ' ')
        lineString = lineString[:-7]
        list.append(lineString)

    return list

def scrape():
    driver = webdriver.Chrome(executable_path="/Users/andreas/desktop/odds/chromedriver")
    urlResultsPl = "https://www.oddsportal.com/soccer/england/premier-league/results/#/page/"
    xPathResultsPl = "//*[@id='tournamentTable']"
    urlStandingsPl = "https://www.oddsportal.com/soccer/england/premier-league/standings/"
    xPathStandingsPl = "//*[@id='table-type-1']"

    urlResultsLaLiga = "https://www.oddsportal.com/soccer/spain/laliga/results/#/page/"
    xPathResultsLaLiga = "//*[@id='tournamentTable']"
    urlStandingsLaLiga = "https://www.oddsportal.com/soccer/spain/laliga/standings/"
    xPathStandingsLaLiga = "//*[@id='table-type-1']"

    urlResultsBundesliga = "https://www.oddsportal.com/soccer/germany/bundesliga/results/#/page/"
    xPathResultsBundesliga = "//*[@id='tournamentTable']"
    urlStandingsBundesliga = "https://www.oddsportal.com/soccer/germany/bundesliga/standings/"
    xPathStandingsBundesliga = "//*[@id='table-type-1']"

    urlResultsSerieA = "https://www.oddsportal.com/soccer/italy/serie-a/results/#/page/"
    xPathResultsSerieA = "//*[@id='tournamentTable']"
    urlStandingsSerieA = "https://www.oddsportal.com/soccer/italy/serie-a/standings/"
    xPathStandingsSerieA = "//*[@id='table-type-1']"


    urlResultsNorge = "https://www.oddsportal.com/soccer/norway/eliteserien/results/#/page/"
    xPathResultsNorge = "//*[@id='tournamentTable']"
    urlStandingsNorge = "https://www.oddsportal.com/soccer/norway/eliteserien/standings/"
    xPathStandingsNorge = "//*[@id='table-type-1']"

    urlResultsChamp = "https://www.oddsportal.com/soccer/england/championship/results/#/page/"
    xPathResultsChamp = "//*[@id='tournamentTable']"
    urlStandingsChamp = "https://www.oddsportal.com/soccer/england/championship/standings/"
    xPathStandingsChamp = "//*[@id='table-type-1']"


    listResults = []
    listStandings = []

    for i in range(1,2):
        driver.get(urlResultsSerieA+str(i))
        time.sleep(5)
        listResults += getResults(xPathResultsSerieA,driver)

    driver.get(urlStandingsSerieA)
    time.sleep(5)
    listStandings = getStandings(xPathStandingsSerieA,driver)

    return listStandings,listResults
