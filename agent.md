# Agent Instructions for Bet-helper Project

This document provides guidelines for AI coding agents working on this project to ensure code quality, prevent common errors, and speed up development.

## Project Overview

**Bet-helper** is a Python web scraping and match prediction tool that:
- Scrapes football match data and odds from OddsPortal using Selenium
- Processes standings and historical match results
- Predicts match outcomes using a naive algorithm
- Outputs predictions to `matches.json`

**Key Files:**
- `main.py` - Main orchestration logic, team/match parsing, prediction algorithm
- `scrape.py` - Web scraping with Selenium (getResults, getStandings, scrapeTodaysMatches)
- `match.py` - Data models (Match, FutureMatch, Team classes)

## Critical: Preventing Indentation and Syntax Errors

### Rule 1: Always Read Full Context Before Editing
**BEFORE making any changes:**
1. Read **at least 30-50 lines** before and after the target area
2. Identify the **complete block structure** (function → for loop → try → if → else)
3. Count indentation levels manually:
   - Module level: 0 spaces
   - Function body: 4 spaces
   - Nested blocks: +4 spaces per level
   - Maximum nesting seen: 5-6 levels deep

### Rule 2: Verify Block Matching
**ALWAYS verify matching pairs:**
- Each `try:` must have an `except:` or `finally:` at the same indentation level
- Each `if:` must have a matching `elif:` or `else:` at the same level (if present)
- Each `for:` must have its body at +4 spaces
- Each `def:` must have its body at +4 spaces

### Rule 3: Small, Incremental Changes
1. Make **one logical change** at a time
2. **Test compilation** after each change: `python3 -m py_compile <file>.py`
3. If multiple fixes needed, do them incrementally and verify each one
4. Never edit multiple unrelated sections in one operation

### Rule 4: Context-Aware Search/Replace
**When using search_replace:**
- Include **enough context** to uniquely identify the location
- Include **at least 3-5 lines before and after** the target
- Verify the **exact indentation** matches (spaces, not tabs)
- If replacing multi-line blocks, ensure the replacement has correct indentation

### Rule 5: Post-Edit Verification
**AFTER every edit:**
```bash
python3 -m py_compile scrape.py
python3 -m py_compile main.py
python3 -m py_compile match.py
```
Fix any syntax errors **immediately** before proceeding.

## Code Style and Conventions

### Python Style Guide
- Follow PEP 8
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: 100 characters (prefer 80 when possible)
- Use descriptive variable names
- Add docstrings for functions

### Naming Conventions
- Functions: `snake_case` (e.g., `scrapeTodaysMatches` → should be `scrape_todays_matches`, but maintain existing style for now)
- Classes: `PascalCase` (e.g., `Match`, `Team`, `FutureMatch`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `REQUIRED_ODDS_COUNT`, `DEFAULT_ODD_PLACEHOLDER`)
- Private helpers: prefix with `_` (e.g., `_normalize_team_name`, `_find_team_in_dict`)

### Error Handling
- Use specific exception types when possible
- Log errors with appropriate levels:
  - `logging.error()` - Critical errors that prevent execution
  - `logging.warning()` - Issues that don't stop execution but should be noted
  - `logging.debug()` - Detailed debugging information
- Always include context in error messages

### Code Organization
- Constants at module level (top of file)
- Helper functions before main functions
- Main functions after helpers
- Imports: standard library → third-party → local imports

## File-Specific Guidelines

### scrape.py
**Critical considerations:**
- **Selenium element access**: Elements become stale after navigation - always extract data before navigating
- **XPath selectors**: Multiple fallback selectors are often needed due to dynamic content
- **Wait times**: Use `WebDriverWait` with explicit conditions instead of fixed `time.sleep()` when possible
- **Error recovery**: Wrap Selenium operations in try/except and provide fallbacks

**Common patterns:**
```python
# Extract data BEFORE navigation
data = element.get_attribute('href')
driver.get(url)  # Navigation makes elements stale

# Use WebDriverWait for dynamic content
WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
    EC.presence_of_element_located((By.XPATH, xpath))
)

# Multiple selector fallbacks
xpath_alternatives = [
    primary_xpath,
    fallback_xpath_1,
    fallback_xpath_2
]
```

### main.py
**Critical considerations:**
- **Team name matching**: Use `_find_team_in_dict()` for fuzzy matching
- **String parsing**: Be robust - handle malformed input gracefully
- **Regex compilation**: Use compiled regex patterns at module level (e.g., `SCORE_PATTERN`)

**Common patterns:**
```python
# Always use helper function for team lookup
team_key = _find_team_in_dict(team_name, teams)
if team_key is None:
    logging.warning(f"Team not found: {team_name}")
    continue

# Validate parsed data before use
if not home_team or not away_team:
    logging.debug("Missing team names")
    continue
```

### match.py
**Critical considerations:**
- Class methods should not mutate input unnecessarily
- Return values instead of mutating instance state when appropriate
- Handle edge cases (division by zero, None values)

## Testing and Validation

### Before Committing Changes
1. **Syntax check**: `python3 -m py_compile <file>.py`
2. **Run linter**: `python3 -m flake8 <file>.py` (if configured)
3. **Manual test**: Run `python3 main.py` and verify:
   - No exceptions thrown
   - Logs show expected behavior
   - `matches.json` is created with valid data

### Debugging Tips
- Check `scrape_debug.log` for detailed execution logs
- HTML debug files are saved when scraping fails (check filenames with timestamps)
- Use `logging.debug()` statements for detailed flow tracking

## Common Pitfalls to Avoid

### 1. Stale Element References
❌ **WRONG:**
```python
row = find_element(...)
driver.get(url)  # Navigation
cells = row.find_elements(...)  # STALE!
```

✅ **CORRECT:**
```python
row = find_element(...)
data = extract_all_data_from_row(row)  # Extract before navigation
driver.get(url)
# Use extracted data, not row element
```

### 2. Incomplete Try/Except Blocks
❌ **WRONG:**
```python
try:
    do_something()
else:  # SyntaxError: else without matching if
```

✅ **CORRECT:**
```python
try:
    do_something()
except Exception as e:
    handle_error(e)
```

### 3. Incorrect Indentation After Search/Replace
❌ **WRONG:**
```python
if condition:
    statement1
statement2  # Wrong indentation - should be inside if
```

✅ **CORRECT:**
```python
if condition:
    statement1
    statement2  # Correct indentation
```

### 4. Using Undefined Variables
❌ **WRONG:**
```python
if some_condition:
    result = calculate()
# Later...
use_result(result)  # NameError if condition was False
```

✅ **CORRECT:**
```python
result = None
if some_condition:
    result = calculate()
if result is not None:
    use_result(result)
```

## Development Workflow

### When Making Changes
1. **Read** the full context (30-50 lines)
2. **Understand** the code structure and indentation
3. **Plan** the change (what needs to be modified)
4. **Execute** the change with correct indentation
5. **Verify** with `python3 -m py_compile`
6. **Test** by running the script if applicable

### When Fixing Bugs
1. **Reproduce** the error (read error message carefully)
2. **Locate** the problematic code (check line numbers)
3. **Read context** around the error location
4. **Identify** the root cause
5. **Fix** incrementally (one change at a time)
6. **Verify** compilation and functionality

### When Refactoring
1. **Preserve** existing functionality
2. **Maintain** same indentation structure
3. **Update** all references if renaming
4. **Test** thoroughly after changes
5. **Document** significant changes

## Project-Specific Patterns

### Scraping Pattern
```python
def scrape_data(driver):
    try:
        # Navigate
        driver.get(url)
        time.sleep(PAGE_LOAD_WAIT)
        
        # Wait for element
        element = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        
        # Extract data
        data = extract_data(element)
        return data
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return None
```

### Team Name Matching Pattern
```python
# Always use fuzzy matching
team_key = _find_team_in_dict(team_name, teams_dict)
if team_key is None:
    logging.warning(f"Team '{team_name}' not found")
    return None
# Use team_key (may be different from team_name)
team_obj = teams_dict[team_key]
```

### Logging Pattern
```python
logging.info(f"Processing {item}")  # High-level flow
logging.debug(f"Detailed info: {variable}")  # Detailed debugging
logging.warning(f"Potential issue: {issue}")  # Non-critical problems
logging.error(f"Error occurred: {error}", exc_info=True)  # Critical errors
```

## Quick Reference

### Indentation Levels in This Project
- **Module level**: 0 spaces
- **Function body**: 4 spaces
- **First nested block**: 8 spaces
- **Second nested block**: 12 spaces
- **Third nested block**: 16 spaces
- **Fourth nested block**: 20 spaces

### Common Commands
```bash
# Syntax check
python3 -m py_compile scrape.py main.py match.py

# Run script
python3 main.py

# Check logs
tail -f scrape_debug.log

# Find syntax errors
python3 -m py_compile *.py 2>&1 | grep -i error
```

### File Structure
```
Bet-helper/
├── main.py           # Main logic, parsing, prediction
├── scrape.py         # Web scraping functions
├── match.py          # Data models
├── agent.md          # This file
├── scrape_debug.log  # Debug logs
└── matches.json      # Output file
```

## Emergency Fixes

If you introduce a syntax error:

1. **Stop** making more changes
2. **Run** `python3 -m py_compile <file>.py` to see exact error
3. **Read** the error message carefully (it shows line number)
4. **Check** the indentation at that line and surrounding lines
5. **Verify** all `try/except`, `if/else`, `for/while` pairs match
6. **Fix** one error at a time, recompiling after each fix

## Notes for Future Development

- The codebase uses **deep nesting** (up to 6 levels) - be extra careful with indentation
- **Selenium operations** are prone to stale elements - always extract data before navigation
- **Team name matching** requires fuzzy logic due to variations (abbreviations, prefixes, etc.)
- **Error handling** is critical - the scraper encounters various edge cases
- **Logging** is extensive - use appropriate log levels for debugging

---

**Remember**: When in doubt, read more context, verify indentation manually, and test compilation frequently. It's better to be slow and correct than fast and broken.
