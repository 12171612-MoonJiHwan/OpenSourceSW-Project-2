import pandas as pd
import numpy as np

# Common
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv', usecols=['year', 'batter_name', 'H', 'avg', 'HR', 'OBP', 'cp', 'R', 'RBI', 'SB', 'war', 'SLG', 'salary'])

# Feature 1
feature_1_year_2015 = df[df['year'] == 2015]
feature_1_year_2016 = df[df['year'] == 2016]
feature_1_year_2017 = df[df['year'] == 2017]
feature_1_year_2018 = df[df['year'] == 2018]

feature_1_data_2015 = feature_1_year_2015[['batter_name', 'H', 'avg', 'HR', 'OBP']]
feature_1_data_2016 = feature_1_year_2016[['batter_name', 'H', 'avg', 'HR', 'OBP']]
feature_1_data_2017 = feature_1_year_2017[['batter_name', 'H', 'avg', 'HR', 'OBP']]
feature_1_data_2018 = feature_1_year_2018[['batter_name', 'H', 'avg', 'HR', 'OBP']]

feature_1_data_2015 = feature_1_data_2015.rename(columns={
    'batter_name': 'player',
    'H': 'hits',
    'avg': 'batting_average',
    'HR': 'homerun',
    'OBP': 'onbase_percentage'
})

feature_1_data_2016 = feature_1_data_2016.rename(columns={
    'batter_name': 'player',
    'H': 'hits',
    'avg': 'batting_average',
    'HR': 'homerun',
    'OBP': 'onbase_percentage'
})

feature_1_data_2017 = feature_1_data_2017.rename(columns={
    'batter_name': 'player',
    'H': 'hits',
    'avg': 'batting_average',
    'HR': 'homerun',
    'OBP': 'onbase_percentage'
})

feature_1_data_2018 = feature_1_data_2018.rename(columns={
    'batter_name': 'player',
    'H': 'hits',
    'avg': 'batting_average',
    'HR': 'homerun',
    'OBP': 'onbase_percentage'
})

print("[Baseball Statistics in 2015]")
for col in ['hits', 'batting_average', 'homerun', 'onbase_percentage']:
    feature_1_rank_2015 = feature_1_data_2015.sort_values(by=col, ascending=False).head(10)
    print(f"\n\n<Top 10 players in {col}>\n")
    print(feature_1_rank_2015[['player', col]])
print("\n=====================================\n")

print("[Baseball Statistics in 2016]")
for col in ['hits', 'batting_average', 'homerun', 'onbase_percentage']:
    feature_1_rank_2016 = feature_1_data_2016.sort_values(by=col, ascending=False).head(10)
    print(f"\n\n<Top 10 players in {col}>\n")
    print(feature_1_rank_2016[['player', col]])
print("\n=====================================\n")

print("[Baseball Statistics in 2017]")
for col in ['hits', 'batting_average', 'homerun', 'onbase_percentage']:
    feature_1_rank_2017 = feature_1_data_2017.sort_values(by=col, ascending=False).head(10)
    print(f"\n\n<Top 10 players in {col}>\n")
    print(feature_1_rank_2017[['player', col]])
print("\n=====================================\n")

print("[Baseball Statistics in 2018]")
for col in ['hits', 'batting_average', 'homerun', 'onbase_percentage']:
    feature_1_rank_2018 = feature_1_data_2018.sort_values(by=col, ascending=False).head(10)
    print(f"\n\n<Top 10 players in {col}>\n")
    print(feature_1_rank_2018[['player', col]])
print("\n=====================================\n")
# End of Feature 1


# Feature 2
feature_2_year = df[df['year'] == 2018]
feature_2_data = feature_2_year[['batter_name', 'war', 'cp']]

feature_2_data = feature_2_data.rename(columns={
    'batter_name': 'player',
    'war': 'winning_assist_rate',
    'cp': 'position'
})

print("[Top player with highest winning assist rate for each position in 2018]")
for position in feature_2_data['position'].unique():
    position_data = feature_2_data[feature_2_data['position'] == position]
    
    for col in ['winning_assist_rate']:
        top_player = feature_2_data.sort_values(by=col, ascending=False).head(1)
        print(f"\nPosition: {position}")
        print(top_player[['player', 'winning_assist_rate', 'position']])
# End of Feature 2


# Feature 3
columns_for_corr = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]

feature_3_corr = columns_for_corr.corr(method='pearson')
corr_with_salary = feature_3_corr['salary'].drop('salary')
corr_highest = corr_with_salary.idxmax()

print("\n\nCorrelations between 'salary' and other variables are:\n")
print(corr_with_salary)
print(f"\nAmong R, H, HR, RBI, SB, war, avg, OBP and SLG,\nThe highest correlation with 'salary' is: {corr_highest}")
# End of Feature 3
