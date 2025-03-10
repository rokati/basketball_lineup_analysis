# Explanation for things

## Data needed
- players from 23-24 season: LeagueDashPlayerStats[0]
- teams from 23-24 season: LeagueDashTeamStats[0]
- all games from 23-24 season: LeagueGameFinder[0]
- all Substitution events from play-by-play data: PlayByPlayV2[0]
- lineup details from 23-24 season: LeagueLineupViz[0]
- overall team details from 23-24 season: TeamPlayerOnOffDetails[0]
- players' on details from 23-24 season: TeamPlayerOnOffDetails[1]
- players' off details from 23-24 season: TeamPlayerOnOffDetails[2]

## NBA Stats API: `LeagueDashTeamStats` Output Explanation

The `LeagueDashTeamStats` endpoint provides **team-level statistics** for a given season. Below is an explanation of each column returned in the API response.

---

### **Basic Team Information**
| Column Name | Description |
|-------------|------------|
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_NAME` | Full team name (e.g., "Los Angeles Lakers") |

---

### **Games & Wins**
| Column Name | Description |
|-------------|------------|
| `GP` | Games played |
| `W` | Wins |
| `L` | Losses |
| `W_PCT` | Win percentage (`W / GP`) |

---

### **Minutes & Scoring**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by the team |
| `PTS` | Total points scored |
| `PLUS_MINUS` | Net points difference when on the court |

---

### **Field Goal Shooting**
| Column Name | Description |
|-------------|------------|
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage (`FGM / FGA`) |

---

### **Three-Point Shooting**
| Column Name | Description |
|-------------|------------|
| `FG3M` | Three-pointers made |
| `FG3A` | Three-pointers attempted |
| `FG3_PCT` | Three-point percentage (`FG3M / FG3A`) |

---

### **Free Throw Shooting**
| Column Name | Description |
|-------------|------------|
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage (`FTM / FTA`) |

---

### **Rebounding**
| Column Name | Description |
|-------------|------------|
| `OREB` | Offensive rebounds |
| `DREB` | Defensive rebounds |
| `REB` | Total rebounds (`OREB + DREB`) |

---

### **Playmaking & Ball Control**
| Column Name | Description |
|-------------|------------|
| `AST` | Assists |
| `TOV` | Turnovers |

---

### **Defense & Fouls**
| Column Name | Description |
|-------------|------------|
| `STL` | Steals |
| `BLK` | Blocks made |
| `BLKA` | Blocks against (how many times the team's shots were blocked) |
| `PF` | Personal fouls committed |
| `PFD` | Personal fouls drawn (fouls received by the team) |

---

### **Team Ranking Metrics**
Each statistic has a **ranking version** that shows how the team compares to others in the league (1 = best, 30 = worst).

| Column Name | Description |
|-------------|------------|
| `GP_RANK` | Rank for games played |
| `W_RANK` | Rank for wins |
| `L_RANK` | Rank for losses |
| `W_PCT_RANK` | Rank for win percentage |
| `MIN_RANK` | Rank for minutes played |
| `FGM_RANK` | Rank for field goals made |
| `FGA_RANK` | Rank for field goals attempted |
| `FG_PCT_RANK` | Rank for field goal percentage |
| `FG3M_RANK` | Rank for three-pointers made |
| `FG3A_RANK` | Rank for three-pointers attempted |
| `FG3_PCT_RANK` | Rank for three-point percentage |
| `FTM_RANK` | Rank for free throws made |
| `FTA_RANK` | Rank for free throws attempted |
| `FT_PCT_RANK` | Rank for free throw percentage |
| `OREB_RANK` | Rank for offensive rebounds |
| `DREB_RANK` | Rank for defensive rebounds |
| `REB_RANK` | Rank for total rebounds |
| `AST_RANK` | Rank for assists |
| `TOV_RANK` | Rank for turnovers |
| `STL_RANK` | Rank for steals |
| `BLK_RANK` | Rank for blocks |
| `BLKA_RANK` | Rank for blocks against |
| `PF_RANK` | Rank for personal fouls committed |
| `PFD_RANK` | Rank for personal fouls drawn |
| `PTS_RANK` | Rank for total points scored |
| `PLUS_MINUS_RANK` | Rank for plus/minus |

---

### **Other Columns**
| Column Name | Description |
|-------------|------------|
| `CFID` | Internal ID (not typically useful) |
| `CFPARAMS` | Internal parameters (not typically useful) |

---


## NBA Stats API: `LeagueGameFinder` Output Explanation

The `LeagueGameFinder` endpoint provides **game-level statistics** for teams or players over a given time range or season. Below is an explanation of each column in the API response.

---

### **Basic Game Information**
| Column Name | Description |
|-------------|------------|
| `SEASON_ID` | The season ID in the format `2XXXXXXX` (e.g., `22023` for the 2023-24 season) |
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., `LAL` for the Los Angeles Lakers) |
| `TEAM_NAME` | Full team name (e.g., "Los Angeles Lakers") |
| `GAME_ID` | Unique identifier for the game |
| `GAME_DATE` | Date of the game in `YYYY-MM-DD` format |
| `MATCHUP` | Shows the opponent and location format (`"LAL vs. BOS"` for home, `"LAL @ BOS"` for away) |
| `WL` | Win (`W`) or Loss (`L`) |

---

### **Minutes & Scoring**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by the team in the game |
| `PTS` | Total points scored |

---

### **Field Goal Shooting**
| Column Name | Description |
|-------------|------------|
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage (`FGM / FGA`) |

---

### **Three-Point Shooting**
| Column Name | Description |
|-------------|------------|
| `FG3M` | Three-pointers made |
| `FG3A` | Three-pointers attempted |
| `FG3_PCT` | Three-point percentage (`FG3M / FG3A`) |

---

### **Free Throw Shooting**
| Column Name | Description |
|-------------|------------|
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage (`FTM / FTA`) |

---

### **Rebounding**
| Column Name | Description |
|-------------|------------|
| `OREB` | Offensive rebounds |
| `DREB` | Defensive rebounds |
| `REB` | Total rebounds (`OREB + DREB`) |

---

### **Playmaking & Ball Control**
| Column Name | Description |
|-------------|------------|
| `AST` | Assists |
| `TOV` | Turnovers |

---

### **Defense & Fouls**
| Column Name | Description |
|-------------|------------|
| `STL` | Steals |
| `BLK` | Blocks |
| `PF` | Personal fouls committed |

---

### **Plus-Minus**
| Column Name | Description |
|-------------|------------|
| `PLUS_MINUS` | Net point differential while on the floor (`Team Points - Opponent Points`) |

---


## NBA Stats API: `PlayByPlayV2` Output Explanation (Substitutions)

The `PlayByPlayV2` endpoint provides **event-level data** for each game, detailing actions such as shots, fouls, turnovers, and substitutions. This document focuses specifically on **substitution events** (`EVENTMSGTYPE == 8`).

---

### **Understanding Substitution Events (`EVENTMSGTYPE == 8`)**

| Column Name | Description |
|-------------|------------|
| `GAME_ID` | Unique identifier for the game |
| `EVENTNUM` | Event number (chronological order within the game) |
| `EVENTMSGTYPE` | Type of event (8 = Substitution) |
| `EVENTMSGACTIONTYPE` | Subtype of event (not always relevant for substitutions) |
| `PERIOD` | Quarter or overtime period in which the substitution occurred |
| `WCTIMESTRING` | Game clock time in **real-world time** (Wall Clock) |
| `PCTIMESTRING` | Game clock time in **period time** (e.g., "8:32" means 8 minutes 32 seconds remaining) |
| `HOMEDESCRIPTION` | Description of the event for the home team (e.g., "LeBron James enters the game") |
| `NEUTRALDESCRIPTION` | Neutral description of the event (often empty) |
| `VISITORDESCRIPTION` | Description of the event for the visiting team |

---

### **Score Information**
| Column Name | Description |
|-------------|------------|
| `SCORE` | Current game score after the event (e.g., "LAL 50 - BOS 48") |
| `SCOREMARGIN` | Difference in score from the leading team’s perspective (e.g., "+2" means leading by 2 points) |

---

### **Player Involved in the Substitution**
#### **Player Checking In**
| Column Name | Description |
|-------------|------------|
| `PERSON1TYPE` | Always `3` for substitutions (indicates the player entering the game) |
| `PLAYER1_ID` | Player ID of the entering player |
| `PLAYER1_NAME` | Name of the entering player |
| `PLAYER1_TEAM_ID` | Team ID of the entering player |
| `PLAYER1_TEAM_CITY` | Team city of the entering player |
| `PLAYER1_TEAM_NICKNAME` | Team nickname of the entering player (e.g., "Lakers") |
| `PLAYER1_TEAM_ABBREVIATION` | Team abbreviation of the entering player (e.g., "LAL") |

---

#### **Player Checking Out**
| Column Name | Description |
|-------------|------------|
| `PERSON2TYPE` | Always `4` for substitutions (indicates the player leaving the game) |
| `PLAYER2_ID` | Player ID of the exiting player |
| `PLAYER2_NAME` | Name of the exiting player |
| `PLAYER2_TEAM_ID` | Team ID of the exiting player |
| `PLAYER2_TEAM_CITY` | Team city of the exiting player |
| `PLAYER2_TEAM_NICKNAME` | Team nickname of the exiting player |
| `PLAYER2_TEAM_ABBREVIATION` | Team abbreviation of the exiting player |

---

### **Other Fields**
| Column Name | Description |
|-------------|------------|
| `PERSON3TYPE` | Not typically used in substitution events |
| `PLAYER3_ID` | Not typically used in substitution events |
| `PLAYER3_NAME` | Not typically used in substitution events |
| `PLAYER3_TEAM_ID` | Not typically used in substitution events |
| `PLAYER3_TEAM_CITY` | Not typically used in substitution events |
| `PLAYER3_TEAM_NICKNAME` | Not typically used in substitution events |
| `PLAYER3_TEAM_ABBREVIATION` | Not typically used in substitution events |
| `VIDEO_AVAILABLE_FLAG` | `1` if a video clip is available for this play, `0` otherwise |

---

## NBA Stats API: `LeagueLineupViz` Output Explanation

The `LeagueLineupViz` endpoint provides **lineup-level statistics** for NBA teams. This includes offensive and defensive ratings, shooting efficiency, pace, and other advanced metrics.

---

### **Lineup Information**
| Column Name | Description |
|-------------|------------|
| `GROUP_ID` | Unique identifier for the lineup (typically a combination of player IDs) |
| `GROUP_NAME` | Names of the players in the lineup (e.g., "LeBron James - Anthony Davis - D'Angelo Russell - Austin Reaves - Jarred Vanderbilt") |
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., `LAL` for the Los Angeles Lakers) |

---

### **Minutes & Efficiency**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by the lineup |

---

### **Advanced Team Ratings**
| Column Name | Description |
|-------------|------------|
| `OFF_RATING` | Offensive rating: points scored per 100 possessions |
| `DEF_RATING` | Defensive rating: points allowed per 100 possessions |
| `NET_RATING` | Net rating: `OFF_RATING - DEF_RATING` (measures overall lineup impact) |
| `PACE` | Estimated number of possessions per 48 minutes |

---

### **Shooting Efficiency**
| Column Name | Description |
|-------------|------------|
| `TS_PCT` | True shooting percentage (`(PTS / (2 * (FGA + 0.44 * FTA)))`), accounts for 2PTs, 3PTs, and FTs |
| `FTA_RATE` | Free throw attempt rate (`FTA / FGA`), measures how often a lineup gets to the free-throw line |

---

### **Team Playmaking & Shot Distribution**
| Column Name | Description |
|-------------|------------|
| `TM_AST_PCT` | Percentage of team field goals that were assisted while this lineup was on the floor |
| `PCT_FGA_2PT` | Percentage of total field goal attempts that were **two-pointers** |
| `PCT_FGA_3PT` | Percentage of total field goal attempts that were **three-pointers** |

---

### **Point Distribution**
| Column Name | Description |
|-------------|------------|
| `PCT_PTS_2PT_MR` | Percentage of total points that came from **mid-range two-pointers** |
| `PCT_PTS_FB` | Percentage of total points that came from **fast-break opportunities** |
| `PCT_PTS_FT` | Percentage of total points that came from **free throws** |
| `PCT_PTS_PAINT` | Percentage of total points that came from **points in the paint** |

---

### **Assist Metrics**
| Column Name | Description |
|-------------|------------|
| `PCT_AST_FGM` | Percentage of made field goals that were **assisted** |
| `PCT_UAST_FGM` | Percentage of made field goals that were **unassisted** |

---

### **Opponent Defensive Metrics**
| Column Name | Description |
|-------------|------------|
| `OPP_FG3_PCT` | Opponent three-point percentage while this lineup was on the floor |
| `OPP_EFG_PCT` | Opponent effective field goal percentage (`(FGM + 0.5 * 3PM) / FGA`) |
| `OPP_FTA_RATE` | Opponent free throw attempt rate (`FTA / FGA`) |
| `OPP_TOV_PCT` | Opponent turnover percentage (how often the opposing team turns the ball over per possession) |

---


## NBA Stats API: `OverallTeamPlayerOnOffDetails` Output Explanation

The `OverallTeamPlayerOnOffDetails` endpoint provides **on/off data** for players and teams. This dataset helps analyze how teams perform when a specific player (or group of players) is on or off the court.

---

### **Grouping Information**
| Column Name | Description |
|-------------|------------|
| `GROUP_SET` | Indicates the group type (e.g., `OnFloor`, `OffFloor`, `Overall`) |
| `GROUP_VALUE` | The specific player(s) or team for which the on/off stats are calculated |

---

### **Team Information**
| Column Name | Description |
|-------------|------------|
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., `BOS` for the Boston Celtics) |
| `TEAM_NAME` | Full name of the team |

---

### **Games & Record**
| Column Name | Description |
|-------------|------------|
| `GP` | Games played where this group (player/team lineup) was tracked |
| `W` | Wins in those games |
| `L` | Losses in those games |
| `W_PCT` | Win percentage (`W / GP`) |

---

### **Minutes & Scoring**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by this group |
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage (`FGM / FGA`) |
| `FG3M` | Three-point field goals made |
| `FG3A` | Three-point field goals attempted |
| `FG3_PCT` | Three-point percentage (`FG3M / FG3A`) |
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage (`FTM / FTA`) |

---

### **Rebounding**
| Column Name | Description |
|-------------|------------|
| `OREB` | Offensive rebounds |
| `DREB` | Defensive rebounds |
| `REB` | Total rebounds (`OREB + DREB`) |

---

### **Playmaking & Turnovers**
| Column Name | Description |
|-------------|------------|
| `AST` | Assists |
| `TOV` | Turnovers |

---

### **Defense & Fouls**
| Column Name | Description |
|-------------|------------|
| `STL` | Steals |
| `BLK` | Blocks |
| `BLKA` | Blocks against (times this group got blocked) |
| `PF` | Personal fouls committed |
| `PFD` | Personal fouls drawn |

---

### **Plus-Minus & Rankings**
| Column Name | Description |
|-------------|------------|
| `PTS` | Points scored |
| `PLUS_MINUS` | Plus-Minus (`Points Scored - Points Allowed` while this group was on the floor) |

Each of the following columns provides the **league-wide rank** for the corresponding stat:

| Rank Columns | Description |
|-------------|------------|
| `GP_RANK` | Rank in games played |
| `W_RANK` | Rank in wins |
| `L_RANK` | Rank in losses |
| `W_PCT_RANK` | Rank in win percentage |
| `MIN_RANK` | Rank in minutes played |
| `FGM_RANK` | Rank in field goals made |
| `FGA_RANK` | Rank in field goals attempted |
| `FG_PCT_RANK` | Rank in field goal percentage |
| `FG3M_RANK` | Rank in three-point field goals made |
| `FG3A_RANK` | Rank in three-point field goals attempted |
| `FG3_PCT_RANK` | Rank in three-point percentage |
| `FTM_RANK` | Rank in free throws made |
| `FTA_RANK` | Rank in free throws attempted |
| `FT_PCT_RANK` | Rank in free throw percentage |
| `OREB_RANK` | Rank in offensive rebounds |
| `DREB_RANK` | Rank in defensive rebounds |
| `REB_RANK` | Rank in total rebounds |
| `AST_RANK` | Rank in assists |
| `TOV_RANK` | Rank in turnovers |
| `STL_RANK` | Rank in steals |
| `BLK_RANK` | Rank in blocks |
| `BLKA_RANK` | Rank in blocks against |
| `PF_RANK` | Rank in personal fouls committed |
| `PFD_RANK` | Rank in personal fouls drawn |
| `PTS_RANK` | Rank in total points scored |
| `PLUS_MINUS_RANK` | Rank in plus-minus |

---

## NBA Stats API: `PlayersOffCourtTeamPlayerOnOffDetails` Output Explanation

The `PlayersOffCourtTeamPlayerOnOffDetails` endpoint provides **on/off data for when a specific player is OFF the court**. This dataset helps analyze how a team performs without a certain player.

---

### **Grouping Information**
| Column Name | Description |
|-------------|------------|
| `GROUP_SET` | Indicates the type of grouping (e.g., `OnFloor`, `OffFloor`, `Overall`) |

---

### **Team Information**
| Column Name | Description |
|-------------|------------|
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., `LAL` for the Los Angeles Lakers) |
| `TEAM_NAME` | Full name of the team |

---

### **Player Off-Court Information**
| Column Name | Description |
|-------------|------------|
| `VS_PLAYER_ID` | The player ID for whom the team's performance is analyzed when they are OFF the court |
| `VS_PLAYER_NAME` | The name of the player who is OFF the court |
| `COURT_STATUS` | Indicates whether the player is ON or OFF the court (for this dataset, it’s `OFF`) |

---

### **Games & Record**
| Column Name | Description |
|-------------|------------|
| `GP` | Games played while this player was OFF the court |
| `W` | Wins in those games |
| `L` | Losses in those games |
| `W_PCT` | Win percentage (`W / GP`) |

---

### **Minutes & Scoring**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by the team when this player was OFF the court |
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage (`FGM / FGA`) |
| `FG3M` | Three-point field goals made |
| `FG3A` | Three-point field goals attempted |
| `FG3_PCT` | Three-point percentage (`FG3M / FG3A`) |
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage (`FTM / FTA`) |

---

### **Rebounding**
| Column Name | Description |
|-------------|------------|
| `OREB` | Offensive rebounds |
| `DREB` | Defensive rebounds |
| `REB` | Total rebounds (`OREB + DREB`) |

---

### **Playmaking & Turnovers**
| Column Name | Description |
|-------------|------------|
| `AST` | Assists |
| `TOV` | Turnovers |

---

### **Defense & Fouls**
| Column Name | Description |
|-------------|------------|
| `STL` | Steals |
| `BLK` | Blocks |
| `BLKA` | Blocks against (times this team got blocked) |
| `PF` | Personal fouls committed |
| `PFD` | Personal fouls drawn |

---

### **Plus-Minus & Rankings**
| Column Name | Description |
|-------------|------------|
| `PTS` | Points scored |
| `PLUS_MINUS` | Plus-Minus (`Points Scored - Points Allowed` while this player was OFF the floor) |

Each of the following columns provides the **league-wide rank** for the corresponding stat:

| Rank Columns | Description |
|-------------|------------|
| `GP_RANK` | Rank in games played |
| `W_RANK` | Rank in wins |
| `L_RANK` | Rank in losses |
| `W_PCT_RANK` | Rank in win percentage |
| `MIN_RANK` | Rank in minutes played |
| `FGM_RANK` | Rank in field goals made |
| `FGA_RANK` | Rank in field goals attempted |
| `FG_PCT_RANK` | Rank in field goal percentage |
| `FG3M_RANK` | Rank in three-point field goals made |
| `FG3A_RANK` | Rank in three-point field goals attempted |
| `FG3_PCT_RANK` | Rank in three-point percentage |
| `FTM_RANK` | Rank in free throws made |
| `FTA_RANK` | Rank in free throws attempted |
| `FT_PCT_RANK` | Rank in free throw percentage |
| `OREB_RANK` | Rank in offensive rebounds |
| `DREB_RANK` | Rank in defensive rebounds |
| `REB_RANK` | Rank in total rebounds |
| `AST_RANK` | Rank in assists |
| `TOV_RANK` | Rank in turnovers |
| `STL_RANK` | Rank in steals |
| `BLK_RANK` | Rank in blocks |
| `BLKA_RANK` | Rank in blocks against |
| `PF_RANK` | Rank in personal fouls committed |
| `PFD_RANK` | Rank in personal fouls drawn |
| `PTS_RANK` | Rank in total points scored |
| `PLUS_MINUS_RANK` | Rank in plus-minus |

---

## NBA Stats API: `PlayersOnCourtTeamPlayerOnOffDetails` Output Explanation

The `PlayersOnCourtTeamPlayerOnOffDetails` endpoint provides **on-court data for when a specific player is ON the court**. This dataset is crucial for understanding the performance of the team when the player is actively playing.

---

### **Grouping Information**
| Column Name | Description |
|-------------|------------|
| `GROUP_SET` | Indicates the type of grouping (e.g., `OnFloor`, `OffFloor`, `Overall`) |

---

### **Team Information**
| Column Name | Description |
|-------------|------------|
| `TEAM_ID` | Unique identifier for the team |
| `TEAM_ABBREVIATION` | Team abbreviation (e.g., `LAL` for the Los Angeles Lakers) |
| `TEAM_NAME` | Full name of the team |

---

### **Player On-Court Information**
| Column Name | Description |
|-------------|------------|
| `VS_PLAYER_ID` | The player ID for whom the team's performance is analyzed when they are ON the court |
| `VS_PLAYER_NAME` | The name of the player who is ON the court |
| `COURT_STATUS` | Indicates whether the player is ON or OFF the court (for this dataset, it’s `ON`) |

---

### **Games & Record**
| Column Name | Description |
|-------------|------------|
| `GP` | Games played while this player was ON the court |
| `W` | Wins in those games |
| `L` | Losses in those games |
| `W_PCT` | Win percentage (`W / GP`) |

---

### **Minutes & Scoring**
| Column Name | Description |
|-------------|------------|
| `MIN` | Total minutes played by the team when this player was ON the court |
| `FGM` | Field goals made |
| `FGA` | Field goals attempted |
| `FG_PCT` | Field goal percentage (`FGM / FGA`) |
| `FG3M` | Three-point field goals made |
| `FG3A` | Three-point field goals attempted |
| `FG3_PCT` | Three-point percentage (`FG3M / FG3A`) |
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `FT_PCT` | Free throw percentage (`FTM / FTA`) |

---

### **Rebounding**
| Column Name | Description |
|-------------|------------|
| `OREB` | Offensive rebounds |
| `DREB` | Defensive rebounds |
| `REB` | Total rebounds (`OREB + DREB`) |

---

### **Playmaking & Turnovers**
| Column Name | Description |
|-------------|------------|
| `AST` | Assists |
| `TOV` | Turnovers |

---

### **Defense & Fouls**
| Column Name | Description |
|-------------|------------|
| `STL` | Steals |
| `BLK` | Blocks |
| `BLKA` | Blocks against (times this team got blocked) |
| `PF` | Personal fouls committed |
| `PFD` | Personal fouls drawn |

---

### **Plus-Minus & Rankings**
| Column Name | Description |
|-------------|------------|
| `PTS` | Points scored |
| `PLUS_MINUS` | Plus-Minus (`Points Scored - Points Allowed` while this player was ON the floor) |

Each of the following columns provides the **league-wide rank** for the corresponding stat:

| Rank Columns | Description |
|-------------|------------|
| `GP_RANK` | Rank in games played |
| `W_RANK` | Rank in wins |
| `L_RANK` | Rank in losses |
| `W_PCT_RANK` | Rank in win percentage |
| `MIN_RANK` | Rank in minutes played |
| `FGM_RANK` | Rank in field goals made |
| `FGA_RANK` | Rank in field goals attempted |
| `FG_PCT_RANK` | Rank in field goal percentage |
| `FG3M_RANK` | Rank in three-point field goals made |
| `FG3A_RANK` | Rank in three-point field goals attempted |
| `FG3_PCT_RANK` | Rank in three-point percentage |
| `FTM_RANK` | Rank in free throws made |
| `FTA_RANK` | Rank in free throws attempted |
| `FT_PCT_RANK` | Rank in free throw percentage |
| `OREB_RANK` | Rank in offensive rebounds |
| `DREB_RANK` | Rank in defensive rebounds |
| `REB_RANK` | Rank in total rebounds |
| `AST_RANK` | Rank in assists |
| `TOV_RANK` | Rank in turnovers |
| `STL_RANK` | Rank in steals |
| `BLK_RANK` | Rank in blocks |
| `BLKA_RANK` | Rank in blocks against |
| `PF_RANK` | Rank in personal fouls committed |
| `PFD_RANK` | Rank in personal fouls drawn |
| `PTS_RANK` | Rank in total points scored |
| `PLUS_MINUS_RANK` | Rank in plus-minus |

---