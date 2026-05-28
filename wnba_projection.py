import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_wnba_projections():
    print("🚀 WNBA Predictive Model v11.9 — Integer Projections\n")
    
    try:
        stats_df = pd.read_csv("wnba-player-stats.csv", skiprows=1)
        print(f"Loaded {len(stats_df)} players from 2026 stats file.")
    except FileNotFoundError:
        print("❌ wnba-player-stats.csv not found.")
        return None
    
    stats_df = stats_df.rename(columns={'Min': 'Minutes', 'REB': 'Total_REB', 'FG%': 'FGpct', 'FT%': 'FTpct', '3P%': '3Ppct'})
    stats_df['Player'] = stats_df['Player'].str.replace("''", "'").str.strip()
    
    roster = ["A'ja Wilson", "Jackie Young", "Chelsea Gray", "Chennedy Carter", "Jewell Loyd", "NaLyssa Smith", "Brianna Turner", "Kierstan Bell", "Janiah Barker", "Dana Evans", "Stephanie Talbot", "Cheyenne Parker-Tyus",
               "Paige Bueckers", "Arike Ogunbowale", "Alysha Clark", "Alanna Smith", "Azzi Fudd", "Odyssey Sims", "Li Yueru", "Maddy Siegrist", "Aziaha James", "Dulcy Fankam Mendjiadeu", "Awak Kuier", "Jessica Shepard",
               "Caitlin Clark", "Aliyah Boston", "Kelsey Mitchell", "Sophie Cunningham", "Myisha Hines-Allen", "Lexie Hull", "Damiris Dantas", "Temi Fagbenle", "Grace Berger", "Monique Billings", "Raven Johnson", "Makayla Timpson",
               "Gabby Williams", "Kayla Thornton", "Kaitlyn Chen", "Veronica Burton", "Laeticia Amihere", "Cecilia Zandalasini", "Kaila Charles", "Justė Jocytė", "Tiffany Hayes", "Kiah Stokes", "Iliana Rupert", "Janelle Salaün"]
    
    data = {
        'Player': roster,
        'Position': ['F','G','G','G','G','F','F','F','F','G','F','F','G','G','F','F','G','G','C','F','G','F','C','F','G','F','G','G','F','G','G','F','G','F','G','F','F','F','G','G','F','F','G','G','G','C','F','F'],
        'Team': ['LVA']*12 + ['DAL']*12 + ['IND']*12 + ['GSV']*12,
        'Opponent': ['DAL']*12 + ['LVA']*12 + ['GSV']*12 + ['IND']*12,
        'Home': [0]*12 + [1]*12 + [0]*12 + [1]*12,
        'Rest_Days': [2]*12 + [1]*12 + [2]*12 + [1]*12,
        'Injury_Risk': [5,6,8,9,7,12,10,15,18,8,11,14] * 4,
    }
    
    df = pd.DataFrame(data)
    df = df.merge(stats_df.drop(columns=['Team'], errors='ignore'), on='Player', how='left')
    df = df.fillna(0)
    
    df['Home_Num'] = df['Home']
    df['Is_Guard'] = df['Position'].str.contains('G')
    
    df['Star_Boost'] = 1.0
    df.loc[df['Player'] == "A'ja Wilson", 'Star_Boost'] = 1.05
    df.loc[df['Player'] == "Caitlin Clark", 'Star_Boost'] = 0.96
    
    df['Guard_Mult'] = np.where(df['Is_Guard'], 0.83, 1.0)
    
    df['Usage_Rate'] = (df['FGA'] / df['Minutes'].replace(0, 30)) * 100
    df['Usage_Rate'] = df['Usage_Rate'].clip(8, 31)
    
    df['Projected_Minutes'] = df['Minutes'] * (1 - df['Injury_Risk']/100) * np.where(df['Rest_Days'] >= 2, 1.00, 0.95)
    df['Projected_Minutes'] = df['Projected_Minutes'].clip(10, 34).round(0).astype(int)
    
    df['Form_5'] = np.random.normal(1.0, 0.028, len(df)).clip(0.94, 1.06)
    
    for col in ['FGA', 'AST', 'OREB', 'DREB', 'STL', 'BLK']:
        if col in df.columns:
            usage_factor = df['Usage_Rate'] / 24.0
            df[col] = round(df[col] * 0.86 + df[col] * df['Form_5'] * 0.14 * usage_factor * df['Guard_Mult'], 2)
    
    team_pace = {'LVA': 98.2, 'DAL': 101.5, 'IND': 100.2, 'GSV': 97.8}
    df['Opp_Pace_Factor'] = df['Opponent'].map(team_pace) / 99.0
    
    for i, row in df.iterrows():
        mult = row['Form_5'] * row['Star_Boost'] * 0.80
        pace = row['Opp_Pace_Factor']
        df.at[i, 'FGA'] = round(row['FGA'] * mult * pace * 0.75, 2)
        df.at[i, 'AST'] = round(row['AST'] * mult * (1.00 if row['Home'] else 0.94), 2)
        df.at[i, 'OREB'] = round(row['OREB'] * mult, 2)
        df.at[i, 'DREB'] = round(row['DREB'] * mult, 2)
        df.at[i, 'STL'] = round(row['STL'] * mult, 2)
        df.at[i, 'BLK'] = round(row['BLK'] * mult * 1.03, 2)
    
    df['FGM'] = df['FGA'] * (df['FGpct'] / 100)
    df['3PM'] = df['3PA'] * (df.get('3Ppct', pd.Series([35.0]*len(df))) / 100)
    df['FTM'] = df['FTA'] * (df['FTpct'] / 100)
    df['2PA'] = df['FGA'] - df['3PA']
    df['2PM'] = df['FGM'] - df['3PM']
    df['PTS'] = (df['FGM'] * 2) + (df['3PM'] * 3) + df['FTM']
    
    df['FanDuel_FP'] = (
        df['PTS'] * 1.0 +
        df['Total_REB'] * 1.2 +
        df['AST'] * 1.5 +
        df['BLK'] * 3.0 +
        df['STL'] * 3.0 -
        df['TO'] * 1.0
    )
    
    # Monte Carlo
    n_sim = 1000
    sims = np.random.normal(0, 3.1, (n_sim, len(df))) * (df['Projected_Minutes'] / 33).values
    sim_fps = df['FanDuel_FP'].values[:, np.newaxis] + sims.T
    
    df['Predicted_FP'] = np.median(sim_fps, axis=1).round(0).astype(int)   # ← Integer Fantasy Points
    
    df['PRA'] = (df['PTS'] + df['Total_REB'] + df['AST']).round(0).astype(int)
    
    df['Home'] = df['Home'].map({1: 'Yes', 0: 'No'})
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    projection_save = df[['Player', 'Team', 'Opponent', 'Projected_Minutes', 'PTS', 'Total_REB', 'AST', 'Predicted_FP', 'PRA']].copy()
    projection_save['Date'] = today
    projection_save.to_csv(f"projections_{today}.csv", index=False)
    
    column_order = ['Player', 'Position', 'Team', 'Opponent', 'Home', 'Rest_Days', 'Injury_Risk', 'Minutes', 'Projected_Minutes', 'PTS', 'FGA', 'FGM', '2PA', '2PM', '3PA', '3PM', 'FTA', 'FTM', 'Total_REB', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TO', 'PRA', 'PR', 'PA', 'RA', 'BS', 'Predicted_FP']
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]
    
    df.to_excel(f'wnba_projections_{today}.xlsx', index=False)
    df.to_csv(f'wnba_projections_{today}.csv', index=False)
    
    print(f"✅ SUCCESS — v11.9 Clean Integer Version ({len(df)} players)")
    print(f"Files saved for {today}\n")
    
    print("🏆 TOP 10 PROJECTED FAN DUEL SCORES")
    print("-" * 65)
    top10_cols = ['Player', 'Team', 'Opponent', 'Home', 'Projected_Minutes', 'PTS', 'Total_REB', 'AST', 'Predicted_FP', 'PRA']
    top10 = df[top10_cols].sort_values('Predicted_FP', ascending=False).head(10)
    print(top10.to_string(index=False))
    
    return df

if __name__ == "__main__":
    generate_wnba_projections()
