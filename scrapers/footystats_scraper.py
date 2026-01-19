"""FootyStats.org scraper implementation."""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from scrapers.base_scraper import BaseScraper
import logging
import time
import re
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it (will use system env vars)
    pass


# Constants
BASE_URL = "https://footystats.org"
PAGE_LOAD_WAIT = 3
ELEMENT_WAIT_TIMEOUT = 10
SCROLL_WAIT = 2


class FootyStatsScraper(BaseScraper):
    """
    FootyStats.org scraper implementation.
    
    This scraper extracts data from footystats.org:
    - League tables from /{country}/{league-name}
    - Fixtures from /{country}/{league-name}/fixtures
    - Detailed fixture data from H2H pages
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, cookies_file: str = ".footystats_cookies.pkl"):
        """
        Initialize the FootyStats scraper with Selenium WebDriver.
        
        Args:
            username: FootyStats username/email (or set FOOTYSTATS_USERNAME env var)
            password: FootyStats password (or set FOOTYSTATS_PASSWORD env var)
            cookies_file: Path to file for saving/loading session cookies
        """
        self.driver = None
        # Dictionary to store H2H stats URLs: key is (home_team, away_team, date), value is URL
        self.h2h_urls = {}
        self._logged_in = False
        self.username = username or os.getenv('FOOTYSTATS_USERNAME')
        self.password = password or os.getenv('FOOTYSTATS_PASSWORD')
        self.cookies_file = cookies_file
    
    def _get_driver(self):
        """Get or create a Chrome WebDriver instance."""
        if self.driver is None:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            options.add_argument('--window-size=1920,1080')
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        return self.driver
    
    def _close_driver(self):
        """Close the WebDriver if it exists."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
    
    def _save_cookies(self):
        """Save current session cookies to file."""
        if self.driver is None:
            return
        
        try:
            cookies = self.driver.get_cookies()
            with open(self.cookies_file, 'wb') as f:
                pickle.dump(cookies, f)
            logging.debug(f"Saved {len(cookies)} cookies to {self.cookies_file}")
        except Exception as e:
            logging.warning(f"Could not save cookies: {e}")
    
    def _load_cookies(self) -> bool:
        """
        Load saved cookies and check if they're still valid.
        
        Returns:
            True if cookies were loaded and are valid, False otherwise
        """
        if not os.path.exists(self.cookies_file):
            logging.debug("No saved cookies found")
            return False
        
        try:
            with open(self.cookies_file, 'rb') as f:
                cookies = pickle.load(f)
            
            if not cookies:
                logging.debug("Cookies file is empty")
                return False
            
            driver = self._get_driver()
            
            # First, navigate to the site to set domain
            driver.get(f"{BASE_URL}/")
            time.sleep(1)
            
            # Load cookies into the driver
            cookies_loaded = 0
            for cookie in cookies:
                try:
                    # Remove 'expiry' field if present (Selenium handles expiry internally)
                    # But check if cookie has expired first
                    if 'expiry' in cookie:
                        if cookie['expiry'] < time.time():
                            logging.debug(f"Cookie {cookie.get('name')} has expired, skipping")
                            continue
                        # Remove expiry for Selenium
                        cookie_copy = cookie.copy()
                        del cookie_copy['expiry']
                        driver.add_cookie(cookie_copy)
                        cookies_loaded += 1
                    else:
                        driver.add_cookie(cookie)
                        cookies_loaded += 1
                except Exception as e:
                    logging.debug(f"Could not add cookie {cookie.get('name')}: {e}")
                    continue
            
            if cookies_loaded == 0:
                logging.debug("No valid cookies could be loaded")
                return False
            
            logging.debug(f"Loaded {cookies_loaded} cookies, validating...")
            
            # Test if cookies are still valid by visiting the site
            driver.get(f"{BASE_URL}/")
            time.sleep(PAGE_LOAD_WAIT)
            
            # Check if we're logged in by looking for logout link or checking current URL
            try:
                # Try to find logout link or user menu
                logout_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/logout') or contains(text(), 'Logout')]")
                if logout_links:
                    logging.info("Successfully loaded valid cookies - already logged in")
                    self._logged_in = True
                    return True
                
                # Alternative: check if we can access a protected page
                # Visit the login page - if logged in, we might be redirected
                driver.get(f"{BASE_URL}/login")
                time.sleep(2)
                
                # If we're redirected away from login page, we're logged in
                if '/login' not in driver.current_url.lower():
                    logging.info("Successfully loaded valid cookies - redirected from login page")
                    self._logged_in = True
                    return True
                
                # If still on login page, cookies are invalid
                logging.debug("Loaded cookies appear to be invalid (still on login page)")
                return False
                
            except Exception as e:
                logging.warning(f"Error validating cookies: {e}")
                return False
            
        except Exception as e:
            logging.warning(f"Could not load cookies: {e}")
            return False
    
    def login(self) -> bool:
        """
        Log in to footystats.org using provided credentials.
        Saves cookies after successful login for future use.
        
        Returns:
            True if login successful, False otherwise
        """
        if self._logged_in:
            logging.debug("Already logged in to footystats.org")
            return True
        
        if not self.username or not self.password:
            logging.warning("No credentials provided for footystats.org login. H2H stats may not be available.")
            return False
        
        driver = self._get_driver()
        
        try:
            logging.info("Logging in to footystats.org...")
            driver.get(f"{BASE_URL}/login")
            time.sleep(PAGE_LOAD_WAIT)
            
            # Find and fill username field
            try:
                username_field = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                    EC.presence_of_element_located((By.ID, "username"))
                )
                username_field.clear()
                username_field.send_keys(self.username)
                logging.debug("Entered username")
            except Exception as e:
                logging.error(f"Could not find username field: {e}")
                return False
            
            # Find and fill password field
            try:
                password_field = driver.find_element(By.ID, "password")
                password_field.clear()
                password_field.send_keys(self.password)
                logging.debug("Entered password")
            except Exception as e:
                logging.error(f"Could not find password field: {e}")
                return False
            
            # Find and click login button
            try:
                login_button = driver.find_element(By.ID, "register_submit")
                login_button.click()
                time.sleep(PAGE_LOAD_WAIT * 2)  # Wait for login to complete and page to redirect
                logging.debug("Clicked login button")
            except Exception as e:
                logging.error(f"Could not click login button: {e}")
                return False
            
            # Verify login success - check if we're redirected away from login page
            current_url = driver.current_url
            if '/login' not in current_url.lower():
                logging.info("Login successful - redirected from login page")
                self._logged_in = True
                # Save cookies after successful login
                self._save_cookies()
                return True
            
            # Check for error messages or if still on login page
            try:
                # Look for error messages
                error_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'error') or contains(@class, 'alert')]")
                if error_elements:
                    error_text = error_elements[0].text
                    logging.error(f"Login failed - error message: {error_text}")
                    return False
                
                # Check if we can find a logout button or user menu (indicates logged in)
                logout_elem = driver.find_elements(By.XPATH, "//a[contains(@href, '/logout') or contains(text(), 'Logout')]")
                if logout_elem:
                    logging.info("Login successful - logout link found")
                    self._logged_in = True
                    # Save cookies after successful login
                    self._save_cookies()
                    return True
            except Exception:
                pass
            
            # If still on login page, login likely failed
            if '/login' in current_url.lower():
                logging.warning("Still on login page after login attempt - credentials may be invalid")
                return False
            
            # Default: assume success if we got here
            logging.info("Login appears successful")
            self._logged_in = True
            # Save cookies after successful login
            self._save_cookies()
            return True
            
        except Exception as e:
            logging.error(f"Error during login: {e}", exc_info=True)
            return False
    
    def scrape_league_table(self, league_path: str) -> List[Dict[str, Any]]:
        """
        Scrape league table/standings data from footystats.org.
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga")
            
        Returns:
            List of dictionaries, each representing a team with structured data:
            {
                "position": 1,
                "team_name": "Barcelona",
                "matches_played": 21,
                "wins": 15,
                "draws": 4,
                "losses": 2,
                "goals_scored": 40,
                "goals_conceded": 14,
                "points": 49
            }
        """
        driver = self._get_driver()
        standings_list = []
        
        try:
            url = f"{BASE_URL}{league_path}"
            logging.info(f"Scraping league table from {url}")
            driver.get(url)
            time.sleep(PAGE_LOAD_WAIT)
            
            # Scroll to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_WAIT)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(SCROLL_WAIT)
            
            # Find the standings table
            xpath_alternatives = [
                "//*[@id='league-tables-wrapper']/div/div[1]/table",
                "//table[@class='full-league-table']",
                "//table[contains(@class, 'table-sort')]",
            ]
            
            standings_table = None
            for xpath_alt in xpath_alternatives:
                try:
                    standings_table = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                        EC.presence_of_element_located((By.XPATH, xpath_alt))
                    )
                    logging.info(f"Found standings table using XPath: {xpath_alt}")
                    break
                except Exception:
                    continue
            
            if standings_table is None:
                logging.error(f"Could not find standings table for {league_path}")
                return standings_list
            
            # Extract all rows from tbody
            rows = standings_table.find_elements(By.XPATH, ".//tbody//tr")
            logging.info(f"Found {len(rows)} rows in standings table")
            
            for row in rows:
                try:
                    # Extract position (td.position span)
                    try:
                        position_elem = row.find_element(By.XPATH, ".//td[@class='position']//span")
                        position_str = position_elem.text.strip()
                    except Exception:
                        position_elem = row.find_element(By.XPATH, ".//td[contains(@class, 'position')]")
                        position_str = position_elem.text.strip()
                    
                    try:
                        position = int(position_str)
                    except ValueError:
                        logging.warning(f"Could not parse position: {position_str}")
                        continue
                    
                    # Extract team name (td.team a)
                    try:
                        team_elem = row.find_element(By.XPATH, ".//td[@class='team']//a")
                        team_name = team_elem.text.strip()
                    except Exception:
                        try:
                            team_elem = row.find_element(By.XPATH, ".//td[contains(@class, 'team')]//a")
                            team_name = team_elem.text.strip()
                        except Exception:
                            logging.warning(f"Could not find team name in row")
                            continue
                    
                    # Extract MP (td.mp)
                    try:
                        mp_elem = row.find_element(By.XPATH, ".//td[@class='mp']")
                        mp_str = mp_elem.text.strip()
                        matches_played = int(mp_str) if mp_str.isdigit() else 0
                    except Exception:
                        matches_played = 0
                    
                    # Extract W (td.win)
                    try:
                        win_elem = row.find_element(By.XPATH, ".//td[@class='win']")
                        wins_str = win_elem.text.strip()
                        wins = int(wins_str) if wins_str.isdigit() else 0
                    except Exception:
                        wins = 0
                    
                    # Extract D (td.draw)
                    try:
                        draw_elem = row.find_element(By.XPATH, ".//td[@class='draw']")
                        draws_str = draw_elem.text.strip()
                        draws = int(draws_str) if draws_str.isdigit() else 0
                    except Exception:
                        draws = 0
                    
                    # Extract L (td.loss)
                    try:
                        loss_elem = row.find_element(By.XPATH, ".//td[@class='loss']")
                        losses_str = loss_elem.text.strip()
                        losses = int(losses_str) if losses_str.isdigit() else 0
                    except Exception:
                        losses = 0
                    
                    # Extract GF (td.gf)
                    try:
                        gf_elem = row.find_element(By.XPATH, ".//td[@class='gf']")
                        gf_str = gf_elem.text.strip()
                        goals_scored = int(gf_str) if gf_str.isdigit() else 0
                    except Exception:
                        goals_scored = 0
                    
                    # Extract GA (td.ga)
                    try:
                        ga_elem = row.find_element(By.XPATH, ".//td[@class='ga']")
                        ga_str = ga_elem.text.strip()
                        goals_conceded = int(ga_str) if ga_str.isdigit() else 0
                    except Exception:
                        goals_conceded = 0
                    
                    # Extract Points (td.points)
                    try:
                        points_elem = row.find_element(By.XPATH, ".//td[@class='points']")
                        points_str = points_elem.text.strip()
                        points = int(points_str) if points_str.isdigit() else 0
                    except Exception:
                        points = 0
                    
                    # Create structured dictionary
                    team_data = {
                        "position": position,
                        "team_name": team_name,
                        "matches_played": matches_played,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "goals_scored": goals_scored,
                        "goals_conceded": goals_conceded,
                        "points": points
                    }
                    standings_list.append(team_data)
                    logging.debug(f"Parsed standings: {team_name} (pos {position}, {points} pts)")
                    
                except Exception as e:
                    logging.warning(f"Error parsing standings row: {e}")
                    continue
            
            logging.info(f"Successfully scraped {len(standings_list)} standings entries")
            
        except Exception as e:
            logging.error(f"Error scraping league table from {league_path}: {e}", exc_info=True)
        
        return standings_list
    
    def scrape_fixtures(
        self,
        league_path: str,
        include_past: bool = True,
        include_future: bool = True,
        min_date: datetime | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Scrape fixture data (both past and future matches) from footystats.org.
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga")
            include_past: Whether to include past/finished fixtures
            include_future: Whether to include future/scheduled fixtures
            min_date: Optional optimization for past-only scrapes. When provided, we will
                iterate date groups from most-recent to oldest and stop once the group date
                is older than min_date. This avoids scanning the whole season when doing
                small delta updates.
            
        Returns:
            List of dictionaries, each representing a date group with matches:
            {
                "date": "08 Jan 2026",
                "matches": [
                    {
                        "time": "21:00",
                        "home_team": "Arsenal",
                        "away_team": "Liverpool",
                        "home_score": 2,  # None for future matches
                        "away_score": 1,  # None for future matches
                        "home_odds": 1.59,  # None if not available
                        "draw_odds": 4.22,  # None if not available
                        "away_odds": 5.60,  # None if not available
                        "h2h_url": "https://..."  # Optional H2H stats URL
                    },
                    ...
                ]
            }
        """
        driver = self._get_driver()
        fixtures_list = []
        
        try:
            url = f"{BASE_URL}{league_path}/fixtures"
            logging.info(f"Scraping fixtures from {url} (include_past={include_past}, include_future={include_future})")
            driver.get(url)
            time.sleep(PAGE_LOAD_WAIT)
            
            # Scroll to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_WAIT)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(SCROLL_WAIT)
            
            # Find all date groups (full-matches-table divs)
            all_date_groups = driver.find_elements(By.XPATH, "//div[contains(@class, 'full-matches-table')]")
            
            # Expand hidden date groups (those with hide-table or dNone classes).
            # Future fixtures are typically in hidden/collapsed sections.
            #
            # Optimization: for past-only delta updates (min_date provided), we avoid expanding
            # hidden groups because we only need the newest finished fixtures, which are typically
            # already visible.
            if not (min_date is not None and include_past and not include_future):
                for group in all_date_groups:
                    class_attr = group.get_attribute('class') or ''
                    if 'hide-table' in class_attr or 'dNone' in class_attr:
                        try:
                            # Try to find and click the header/date element to expand
                            # Many collapsible sections expand when you click the header
                            try:
                                header = group.find_element(By.XPATH, ".//h2[contains(@class, 'fs11e')] | .//h2")
                                # Scroll element into view before clicking
                                driver.execute_script(
                                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", header
                                )
                                time.sleep(0.3)
                                header.click()
                                time.sleep(0.5)  # Wait for expansion animation
                                logging.debug(f"Clicked to expand hidden date group")
                            except Exception:
                                # Alternative: Use JavaScript to remove hide classes if clicking doesn't work
                                try:
                                    driver.execute_script(
                                        "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", group
                                    )
                                    time.sleep(0.3)
                                    driver.execute_script("arguments[0].classList.remove('hide-table', 'dNone');", group)
                                    time.sleep(0.5)
                                    logging.debug(f"Expanded hidden date group via JavaScript")
                                except Exception as e:
                                    logging.debug(f"Could not expand date group: {e}")
                                    continue
                        except Exception as e:
                            logging.debug(f"Error expanding date group: {e}")
                            continue
            
            # Now get all date groups again (some may have been expanded)
            date_groups = driver.find_elements(By.XPATH, "//div[contains(@class, 'full-matches-table')]")
            logging.info(f"Found {len(date_groups)} date groups (after expanding hidden ones)")
            
            # Ordering:
            # - Default behavior keeps historical order (start of season -> end) as before.
            # - For past-only delta updates (min_date provided), we keep the DOM order
            #   (typically newest -> oldest) so we can early-break once we reach older dates.
            if not (min_date is not None and include_past and not include_future):
                # Reverse order to process from bottom (start of season) to top (end of season)
                date_groups = list(reversed(date_groups))
                logging.debug(f"Reversed date groups order - processing from start of season to end")
            else:
                logging.info(
                    f"min_date optimization enabled for past-only scrape: will stop when date group < {min_date.strftime('%d %b %Y')}"
                )
            
            for date_group in date_groups:
                try:
                    # Skip if still hidden after expansion attempts (content might not be accessible)
                    class_attr = date_group.get_attribute('class') or ''
                    if 'hide-table' in class_attr and 'dNone' in class_attr:
                        # Try one more time with JavaScript
                        try:
                            driver.execute_script("arguments[0].classList.remove('hide-table', 'dNone');", date_group)
                            time.sleep(0.3)
                        except:
                            logging.debug(f"Skipping date group that couldn't be expanded")
                            continue
                    
                    # Extract gameweek number from data-game-week attribute
                    gameweek = None
                    try:
                        gameweek_attr = date_group.get_attribute('data-game-week')
                        if gameweek_attr:
                            gameweek = int(gameweek_attr)
                            logging.debug(f"Found gameweek {gameweek} for date group")
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Could not parse gameweek: {e}")
                    
                    # Extract date from h2 header (format: "Jan 4 ~" or "May 24 ~")
                    try:
                        date_header = date_group.find_element(By.XPATH, ".//h2[contains(@class, 'fs11e')]")
                        date_text_raw = date_header.text.strip().replace('~', '').strip()
                    except Exception:
                        # Try alternative selector
                        try:
                            date_header = date_group.find_element(By.XPATH, ".//h2")
                            date_text_raw = date_header.text.strip().replace('~', '').strip()
                        except Exception:
                            logging.debug(f"Could not find date header, skipping date group")
                            continue
                    
                    # Skip if date text is empty
                    if not date_text_raw:
                        logging.debug(f"Empty date text, skipping date group")
                        continue
                    
                    # Parse date (format: "Jan 4" or "May 24")
                    # Need to add current year if not present
                    try:
                        # Try parsing with current year
                        current_year = datetime.now().year
                        today = datetime.now().date()
                        date_parts = date_text_raw.split()
                        
                        if len(date_parts) == 2:
                            # Format: "Jan 4" -> "04 Jan YYYY"
                            parsed_date = datetime.strptime(f"{date_text_raw} {current_year}", "%b %d %Y")
                        elif len(date_parts) == 3:
                            # Already has year: "Jan 4 2026"
                            parsed_date = datetime.strptime(date_text_raw, "%b %d %Y")
                        else:
                            logging.warning(f"Unexpected date format: '{date_text_raw}' (parts: {date_parts}), skipping")
                            continue

                        # Season-aware year correction.
                        # Footystats often omits the year in headers; around January this can mis-label
                        # Sep–Dec matches as next-year. We correct based on the scrape mode:
                        # - Past-only scrape: if parsed date is in the future, it's almost certainly last year.
                        # - Future-only scrape: if parsed date is in the past, it's almost certainly next year.
                        if len(date_parts) == 2:
                            if include_past and not include_future and parsed_date.date() > today:
                                parsed_date = parsed_date.replace(year=parsed_date.year - 1)
                            elif include_future and not include_past and parsed_date.date() < today:
                                parsed_date = parsed_date.replace(year=parsed_date.year + 1)
                        
                        # Format as "DD MMM YYYY" (without "1 X 2" suffix)
                        date_formatted = parsed_date.strftime("%d %b %Y")
                    except ValueError as e:
                        logging.warning(f"Could not parse date '{date_text_raw}': {e}, skipping date group")
                        continue

                    # Early stop for delta updates: once we hit older-than-min_date groups,
                    # all remaining groups should be even older (when iterating newest->oldest).
                    if min_date is not None and include_past and not include_future and parsed_date < min_date:
                        logging.info(
                            f"Stopping fixtures scrape early: reached {date_formatted} which is older than min_date {min_date.strftime('%d %b %Y')}"
                        )
                        break
                    
                    # Find all matches in this date group
                    matches = date_group.find_elements(By.XPATH, ".//ul[contains(@class, 'match')]")
                    logging.debug(f"Found {len(matches)} matches for date {date_formatted}")
                    
                    date_matches = []
                    
                    for match_row in matches:
                        try:
                            # FIRST: Try to extract score - if a score exists, it's definitely a past fixture
                            home_score = None
                            away_score = None
                            has_score = False
                            
                            try:
                                score_elem = match_row.find_element(By.XPATH, ".//span[contains(@class, 'ft-score')]")
                                score_text = score_elem.text.strip()
                                # Format: "1 - 1" or "2-0"
                                score_text = score_text.replace(' ', '')
                                if '-' in score_text:
                                    parts = score_text.split('-')
                                    if len(parts) == 2:
                                        try:
                                            home_score = int(parts[0].strip())
                                            away_score = int(parts[1].strip())
                                            has_score = True
                                        except ValueError:
                                            pass
                            except Exception:
                                # Score not found, try alternative selectors
                                try:
                                    # Try finding score in other possible locations
                                    score_elements = match_row.find_elements(By.XPATH, ".//span[contains(@class, 'score')] | .//div[contains(@class, 'score')]")
                                    for score_elem in score_elements:
                                        score_text = score_elem.text.strip()
                                        if score_text and ('-' in score_text or ':' in score_text):
                                            # Try parsing as "X-Y" or "X:Y"
                                            for separator in ['-', ':', '–', '—']:
                                                if separator in score_text:
                                                    parts = score_text.split(separator)
                                                    if len(parts) == 2:
                                                        try:
                                                            home_score = int(parts[0].strip())
                                                            away_score = int(parts[1].strip())
                                                            has_score = True
                                                            break
                                                        except ValueError:
                                                            continue
                                            if has_score:
                                                break
                                except Exception:
                                    pass
                            
                            # Get time_elem (needed for both time extraction and status checking)
                            time_elem = match_row.find_element(By.XPATH, ".//li[contains(@class, 'time')]")
                            
                            # Determine if match is past or future
                            # If fixture has a result/score, it's definitely a past fixture
                            if has_score:
                                is_finished = True
                                is_future = False
                            else:
                                # Fall back to status attributes only if no score found
                                match_status = match_row.get_attribute("data-match-status") or ""
                            status_attr = time_elem.get_attribute("data-match-status") or ""
                            
                            is_finished = status_attr == "complete" or "FT" in match_row.text
                            is_future = status_attr == "incomplete" or "Starts" in match_row.text or "match-time-soon" in match_row.get_attribute("class") or ""
                            
                            # Filter based on include_past/include_future
                            if is_finished and not include_past:
                                continue
                            if is_future and not include_future:
                                continue
                            
                            # Extract time
                            try:
                                time_span = time_elem.find_element(By.XPATH, ".//span[contains(@class, 'timezone-convert')]")
                                time_text = time_span.text.strip()
                                # Parse time from formats like "May 24, 2:00am" or "Sun 11, 4:15pm"
                                # Extract just the time part
                                if ',' in time_text:
                                    time_part = time_text.split(',')[1].strip()
                                    # Format: "2:00am" -> "02:00" or "4:15pm" -> "16:15"
                                    try:
                                        time_obj = datetime.strptime(time_part, "%I:%M%p")
                                        match_time = time_obj.strftime("%H:%M")
                                    except ValueError:
                                        match_time = "00:00"
                                else:
                                    match_time = "00:00"
                            except Exception:
                                match_time = "00:00"
                            
                            # Extract home team
                            try:
                                home_team_elem = match_row.find_element(By.XPATH, ".//a[contains(@class, 'team') and contains(@class, 'home')]")
                                home_team_span = home_team_elem.find_element(By.XPATH, ".//span[contains(@class, 'hover-modal-ajax-team')]")
                                home_team = home_team_span.text.strip()
                            except Exception:
                                logging.warning("Could not extract home team")
                                continue
                            
                            # Extract away team
                            try:
                                away_team_elem = match_row.find_element(By.XPATH, ".//a[contains(@class, 'team') and contains(@class, 'away')]")
                                away_team_span = away_team_elem.find_element(By.XPATH, ".//span[contains(@class, 'hover-modal-ajax-team')]")
                                away_team = away_team_span.text.strip()
                            except Exception:
                                logging.warning("Could not extract away team")
                                continue
                            
                            # Extract H2H stats link
                            h2h_url = None
                            try:
                                h2h_link_elem = match_row.find_element(By.XPATH, ".//a[contains(@class, 'h2h-link')]")
                                h2h_path = h2h_link_elem.get_attribute('href')
                                if h2h_path:
                                    # Make sure it's a full URL
                                    if h2h_path.startswith('/'):
                                        h2h_url = f"{BASE_URL}{h2h_path}"
                                    elif h2h_path.startswith('http'):
                                        h2h_url = h2h_path
                                    else:
                                        h2h_url = f"{BASE_URL}/{h2h_path}"
                                    # Store H2H URL with match identifier
                                    # Store H2H URL with match identifier (use date without "1 X 2")
                                    date_key = date_formatted  # Already formatted as "DD MMM YYYY"
                                    match_key = (home_team, away_team, date_key)
                                    self.h2h_urls[match_key] = h2h_url
                                    logging.debug(f"Stored H2H URL for {home_team} vs {away_team}: {h2h_url}")
                            except Exception as e:
                                logging.debug(f"Could not extract H2H link: {e}")
                                # Continue without H2H URL
                            
                            # Score already extracted above - home_score and away_score are already set
                            # (or None if no score was found)
                            
                            # Extract odds
                            home_odds = None
                            draw_odds = None
                            away_odds = None
                            try:
                                odds_container = match_row.find_element(By.XPATH, ".//div[contains(@class, 'stat') and contains(@class, 'odds')]")
                                odds_spans = odds_container.find_elements(By.XPATH, ".//span[contains(@class, 'col-lg-4')]")
                                
                                for i, odds_span in enumerate(odds_spans[:3]):
                                    odds_text = odds_span.text.strip()
                                    # Extract just the number (remove any extra text)
                                    if odds_text:
                                        # Try to extract decimal number
                                        match = re.search(r'(\d+\.\d+)', odds_text)
                                        if match:
                                            try:
                                                odd_value = float(match.group(1))
                                                if i == 0:
                                                    home_odds = odd_value
                                                elif i == 1:
                                                    draw_odds = odd_value
                                                elif i == 2:
                                                    away_odds = odd_value
                                            except ValueError:
                                                pass
                                        elif odds_text.replace('.', '', 1).isdigit():
                                            try:
                                                odd_value = float(odds_text)
                                                if i == 0:
                                                    home_odds = odd_value
                                                elif i == 1:
                                                    draw_odds = odd_value
                                                elif i == 2:
                                                    away_odds = odd_value
                                            except ValueError:
                                                pass
                            except Exception as e:
                                logging.debug(f"Could not extract odds for {home_team} vs {away_team}: {e}")
                                pass
                            
                            # Log if odds are missing (especially for future matches)
                            if home_odds is None and away_odds is None and draw_odds is None:
                                logging.debug(f"No odds found for {home_team} vs {away_team} (future match or odds not available)")
                            
                            # Get H2H URL if stored
                            match_key = (home_team, away_team, date_formatted)
                            stored_h2h_url = self.h2h_urls.get(match_key)
                            
                            # Create structured match dictionary
                            match_data = {
                                "time": match_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_score": home_score,
                                "away_score": away_score,
                                "home_odds": home_odds,
                                "draw_odds": draw_odds,
                                "away_odds": away_odds,
                                "h2h_url": stored_h2h_url
                            }
                            date_matches.append(match_data)
                            logging.debug(f"Parsed match: {home_team} vs {away_team} at {match_time}")
                            
                        except Exception as e:
                            logging.warning(f"Error parsing match row: {e}")
                            continue
                    
                    # Add date group if it has matches
                    if date_matches:
                        date_group_data = {
                            "date": date_formatted,
                            "gameweek": gameweek,  # Include gameweek number
                            "matches": date_matches
                        }
                        fixtures_list.append(date_group_data)
                        logging.debug(f"Added date group with {len(date_matches)} matches for {date_formatted} (gameweek {gameweek})")
                    
                except Exception as e:
                    logging.warning(f"Error parsing date group: {e}")
                    continue
            
            logging.info(f"Successfully scraped {len(fixtures_list)} date groups with fixtures")
            
        except Exception as e:
            logging.error(f"Error scraping fixtures from {league_path}: {e}", exc_info=True)
        
        return fixtures_list


    def get_newest_finished_fixture_date(self, league_path: str) -> Optional[str]:
        """
        Fast path to detect newest finished match date on the fixtures page.
        Checks all date groups and returns the newest date that has finished matches.
        Returns date formatted as "DD MMM YYYY" or None.
        """
        driver = self._get_driver()
        url = f"{BASE_URL}{league_path}/fixtures"
        logging.info(f"Fast-check newest finished fixture date from {url}")
        driver.get(url)
        time.sleep(PAGE_LOAD_WAIT)

        try:
            WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'full-matches-table')]"))
            )
        except Exception:
            return None

        date_groups = driver.find_elements(By.XPATH, "//div[contains(@class, 'full-matches-table')]")
        if not date_groups:
            return None

        today = datetime.now().date()
        current_year = datetime.now().year
        finished_dates: list[tuple[datetime, str]] = []  # List of (parsed_date, date_formatted)

        # Scan all date groups and collect dates with finished matches
        for date_group in date_groups:
            try:
                # Extract date header text
                try:
                    date_header = date_group.find_element(By.XPATH, ".//h2[contains(@class, 'fs11e')]")
                    date_text_raw = date_header.text.strip().replace('~', '').strip()
                except Exception:
                    try:
                        date_header = date_group.find_element(By.XPATH, ".//h2")
                        date_text_raw = date_header.text.strip().replace('~', '').strip()
                    except Exception:
                        date_text_raw = None

                if not date_text_raw:
                    continue

                # Parse date like existing logic (handles missing year)
                date_parts = date_text_raw.split()
                if len(date_parts) == 2:
                    parsed_date = datetime.strptime(f"{date_text_raw} {current_year}", "%b %d %Y")
                    # If header omitted year and we are in January, top groups can be future season end.
                    # For the *newest finished* match we expect a date <= today; adjust if future.
                    if parsed_date.date() > today:
                        parsed_date = parsed_date.replace(year=parsed_date.year - 1)
                elif len(date_parts) == 3:
                    parsed_date = datetime.strptime(date_text_raw, "%b %d %Y")
                else:
                    continue

                date_formatted = parsed_date.strftime("%d %b %Y")

                # Check individual matches and extract their actual dates (not just the group header)
                matches = date_group.find_elements(By.XPATH, ".//ul[contains(@class, 'match')]")
                if not matches:
                    continue

                # Collect dates from individual finished matches
                match_finished_dates: list[datetime] = []
                for match_row in matches:
                    try:
                        # Check if match is finished
                        is_finished = False
                        try:
                            score_elem = match_row.find_element(By.XPATH, ".//span[contains(@class, 'ft-score')]")
                            score_text = score_elem.text.strip().replace(" ", "")
                            if score_text and ("-" in score_text or ":" in score_text):
                                is_finished = True
                        except Exception:
                            try:
                                time_elem = match_row.find_element(By.XPATH, ".//li[contains(@class, 'time')]")
                                status_attr = time_elem.get_attribute("data-match-status") or ""
                                if status_attr == "complete":
                                    is_finished = True
                            except Exception:
                                continue
                        
                        if not is_finished:
                            continue
                        
                        # Extract the actual match date/time from data-time attribute
                        try:
                            time_elem = match_row.find_element(By.XPATH, ".//li[contains(@class, 'time')]")
                            # Use data-time attribute (Unix timestamp) - most reliable
                            data_time = time_elem.get_attribute("data-time")
                            if data_time:
                                try:
                                    match_dt = datetime.fromtimestamp(int(data_time))
                                    match_finished_dates.append(match_dt)
                                    continue
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        continue
                
                # If this group has finished matches, use the newest match date from the group
                if match_finished_dates:
                    newest_match_date = max(match_finished_dates)
                    date_formatted = newest_match_date.strftime("%d %b %Y")
                    finished_dates.append((newest_match_date, date_formatted))
            except Exception as e:
                logging.debug(f"Error processing date group: {e}")
                continue

        # Return the newest date (most recent) with finished matches
        if finished_dates:
            finished_dates.sort(key=lambda x: x[0], reverse=True)  # Sort by date, newest first
            newest_date = finished_dates[0][1]
            logging.info(f"Found newest finished match date: {newest_date} (checked {len(date_groups)} date groups, found {len(finished_dates)} with finished matches)")
            return newest_date

        return None
    
    def scrape_future_fixtures(self, league_path: str) -> List[Dict[str, Any]]:
        """
        Scrape future/scheduled fixtures only from footystats.org.
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga")
            
        Returns:
            List of dicts with match data: {"home_team": str, "away_team": str, "home_odds": float, "draw_odds": float, "away_odds": float}
        """
        matches_list = []
        
        try:
            # Use scrape_fixtures with include_past=False to get only future fixtures
            fixtures = self.scrape_fixtures(league_path, include_past=False, include_future=True)
            
            # Convert structured fixture data to dicts
            # Each fixture group: {"date": "...", "matches": [...]}
            for date_group in fixtures:
                if "matches" not in date_group or not date_group["matches"]:
                    continue
                
                # Process each match in the date group
                for match_data in date_group["matches"]:
                    try:
                        home_team = match_data.get("home_team")
                        away_team = match_data.get("away_team")
                        
                        if not home_team or not away_team:
                            logging.warning(f"Missing team names in match data")
                            continue
                        
                        # Get odds
                        h_odds = match_data.get("home_odds") or 0.0
                        u_odds = match_data.get("draw_odds") or 0.0
                        b_odds = match_data.get("away_odds") or 0.0
                        
                        # Create dict
                        match_dict = {
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_odds": float(h_odds) if h_odds else 0.0,
                            "draw_odds": float(u_odds) if u_odds else 0.0,
                            "away_odds": float(b_odds) if b_odds else 0.0,
                        }
                        matches_list.append(match_dict)
                        logging.debug(f"Created future match: {home_team} vs {away_team}")
                        
                    except Exception as e:
                        logging.warning(f"Error creating future match from match data: {e}")
                        continue
            
            logging.info(f"Successfully created {len(matches_list)} future match dicts")
            
        except Exception as e:
            logging.error(f"Error scraping future fixtures from {league_path}: {e}", exc_info=True)
        
        return matches_list
    
    def scrape_fixture_details(self, fixture_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed fixture data from H2H/stats page.
        Automatically attempts to load saved cookies or login if needed.
        
        Args:
            fixture_url: Full URL to the fixture details page
                        (e.g., "https://footystats.org/spain/real-club-deportivo-mallorca-vs-girona-fc-h2h-stats#8200733")
            
        Returns:
            Dictionary containing detailed fixture statistics, or None if scraping fails.
            Format:
            {
                "home_team": "Team Name",
                "away_team": "Team Name",
                "possession": {"home": "39%", "away": "61%"},
                "shots": {"home": "16", "away": "5"},
                ...
            }
        """
        driver = self._get_driver()
        details = {}
        
        try:
            # Try to load saved cookies first (if not already logged in)
            if not self._logged_in:
                logging.debug("Not logged in, attempting to load saved cookies...")
                if not self._load_cookies():
                    # Cookies invalid or missing, try to login
                    logging.debug("Cookies invalid or missing, attempting login...")
                    if not self.login():
                        logging.warning("Login failed - H2H stats may not be available")
            
            logging.info(f"Scraping fixture details from {fixture_url}")
            driver.get(fixture_url)
            time.sleep(PAGE_LOAD_WAIT)
            
            # Check if we were redirected to login page (session expired or not logged in)
            if '/login' in driver.current_url.lower():
                logging.info("Redirected to login page, cookies may have expired - attempting login...")
                # Clear invalid cookies
                try:
                    if os.path.exists(self.cookies_file):
                        os.remove(self.cookies_file)
                        logging.debug("Removed invalid cookies file")
                except Exception as e:
                    logging.debug(f"Could not remove cookies file: {e}")
                
                if self.login():
                    # Retry after login
                    driver.get(fixture_url)
                    time.sleep(PAGE_LOAD_WAIT)
                else:
                    logging.warning("Login failed, H2H stats not available for this match")
                    return None
            
            # Scroll to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_WAIT)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(SCROLL_WAIT)
            
            # Find the comparison table
            xpath_alternatives = [
                "//table[contains(@class, 'comparison-table-table')]",
                "//table[contains(@class, 'comparison-table')]",
            ]
            
            comparison_table = None
            for xpath_alt in xpath_alternatives:
                try:
                    comparison_table = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                        EC.presence_of_element_located((By.XPATH, xpath_alt))
                    )
                    logging.info(f"Found comparison table using XPath: {xpath_alt}")
                    break
                except Exception:
                    continue
            
            if comparison_table is None:
                # Check if we're on a login/upgrade page (might be behind paywall)
                page_text = driver.page_source.lower()
                if 'login' in page_text or 'sign up' in page_text or 'upgrade' in page_text or 'premium' in page_text:
                    logging.debug(f"H2H stats page appears to be behind paywall/login at {fixture_url}")
                else:
                    logging.warning(f"Could not find comparison table at {fixture_url} (page may be behind paywall or unavailable)")
                return None
            
            # Extract team names from thead
            try:
                header_row = comparison_table.find_element(By.XPATH, ".//thead//tr")
                header_cells = header_row.find_elements(By.XPATH, ".//th[contains(@class, 'item stat')]")
                
                if len(header_cells) >= 2:
                    # First team (home team)
                    home_team_link = header_cells[0].find_element(By.XPATH, ".//a")
                    home_team = home_team_link.text.strip()
                    details["home_team"] = home_team
                    
                    # Second team (away team)
                    away_team_link = header_cells[1].find_element(By.XPATH, ".//a")
                    away_team = away_team_link.text.strip()
                    details["away_team"] = away_team
                    
                    logging.debug(f"Extracted teams: {home_team} vs {away_team}")
                else:
                    logging.warning("Could not find team names in comparison table header")
                    return None
            except Exception as e:
                logging.warning(f"Error extracting team names: {e}")
                return None
            
            # Extract stats from tbody rows
            try:
                stat_rows = comparison_table.find_elements(By.XPATH, ".//tbody//tr")
                logging.debug(f"Found {len(stat_rows)} stat rows")
                
                for row in stat_rows:
                    try:
                        # Extract stat name (first td with class 'item key')
                        stat_name_elem = row.find_element(By.XPATH, ".//td[contains(@class, 'item key')]")
                        stat_name = stat_name_elem.text.strip().lower().replace(' ', '_')
                        
                        # Extract stat values (two td elements with class 'item stat')
                        stat_cells = row.find_elements(By.XPATH, ".//td[contains(@class, 'item stat')]")
                        
                        if len(stat_cells) >= 2:
                            home_value = stat_cells[0].text.strip()
                            away_value = stat_cells[1].text.strip()
                            
                            # Store as nested dictionary
                            details[stat_name] = {
                                "home": home_value,
                                "away": away_value
                            }
                            logging.debug(f"Extracted {stat_name}: home={home_value}, away={away_value}")
                        else:
                            logging.debug(f"Skipping row with insufficient stat cells: {stat_name}")
                            continue
                            
                    except Exception as e:
                        logging.debug(f"Error parsing stat row: {e}")
                        continue
                
                logging.info(f"Successfully extracted {len([k for k in details.keys() if k not in ['home_team', 'away_team']])} stats")
                
            except Exception as e:
                logging.warning(f"Error extracting stats from comparison table: {e}")
                return None
            
        except Exception as e:
            logging.error(f"Error scraping fixture details from {fixture_url}: {e}", exc_info=True)
            return None
        
        return details if details else None
    
    def get_h2h_url(self, home_team: str, away_team: str, date: str = None) -> Optional[str]:
        """
        Get the H2H stats URL for a specific match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            date: Optional date string (format: "DD MMM YYYY 1 X 2") for more precise matching
            
        Returns:
            H2H stats URL if found, None otherwise
        """
        if date:
            key = (home_team, away_team, date)
            return self.h2h_urls.get(key)
        else:
            # Try to find any match with these teams (less precise)
            for (h, a, d), url in self.h2h_urls.items():
                if h == home_team and a == away_team:
                    return url
        return None
    
    def clear_h2h_urls(self):
        """Clear stored H2H URLs (useful when scraping a new league)."""
        self.h2h_urls.clear()
    
    def __del__(self):
        """Cleanup: close driver when scraper is destroyed."""
        self._close_driver()