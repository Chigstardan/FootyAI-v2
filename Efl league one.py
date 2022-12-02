from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection  import train_test_split, TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.feature_selection  import RFECV, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from scipy.stats import poisson
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
df1 = pd.read_csv("england-efl-league-one-matches-2019-to-2020-stats.csv")
df2 = pd.read_csv("england-efl-league-one-matches-2020-to-2021-stats.csv")
df3 = pd.read_csv('england-efl-league-one-matches-2021-to-2022-stats.csv')
df4 = pd.read_csv('england-efl-league-one-matches-2022-to-2023-stats.csv')
df = pd.concat([df1, df2, df3, df4], ignore_index=True, axis=0)
df['Date'] = df['date_GMT'].astype('datetime64[ns]')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df[(df['Date'] < '2022/10/05')]
df = df.drop(['Game Week', 'referee', 'home_ppg', 'away_ppg',
'attendance', 'date_GMT', 'stadium_name', 'home_team_goal_timings',
'away_team_goal_timings', 'total_goals_at_half_time', 'home_team_goal_timings',
'away_team_goal_timings', 'home_team_corner_count','away_team_corner_count','home_team_yellow_cards', 
'home_team_red_cards', 'away_team_yellow_cards',
'away_team_red_cards', 'home_team_first_half_cards', 
'home_team_second_half_cards', 'away_team_first_half_cards',
'away_team_second_half_cards'], axis=1)
df.drop(df.index[df['status'] == 'suspended'], inplace=True)
net_goals = df.home_team_goal_count - df.away_team_goal_count
f = df.home_team_goal_count == 0
h = df.away_team_goal_count == 0
df['ftsH'] = [1 if i==0 else 0 for i in f]
df['ftsA'] = [1 if i==0 else 0 for i in h]
df['btsH'] = [1 if i==1 else 0 for i in f]
df['btsA'] = [1 if i==1 else 0 for i in h]
df['hgd'] = df.home_team_goal_count - df.away_team_goal_count
df['agd'] = df.away_team_goal_count - df.home_team_goal_count
df['ftsh'] = df.groupby('home_team_name')['ftsH'].apply(lambda x: x.ewm(span=5).mean().shift())
df['ftsa'] = df.groupby('away_team_name')['ftsA'].apply(lambda x: x.ewm(span=5).mean().shift())
df['btsh'] = df.groupby('home_team_name')['btsH'].apply(lambda x: x.ewm(span=5).mean().shift())
df['btsa'] = df.groupby('away_team_name')['btsA'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avgh'] = df.groupby('home_team_name')['home_team_goal_count'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avga'] = df.groupby('away_team_name')['away_team_goal_count'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avghht'] = df.groupby('home_team_name')['home_team_goal_count_half_time'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avgaht'] = df.groupby('away_team_name')['away_team_goal_count_half_time'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avcgh'] = df.groupby('home_team_name')['away_team_goal_count'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avcga'] = df.groupby('away_team_name')['home_team_goal_count'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avh_xg'] = df.groupby('home_team_name')['team_a_xg'].apply(lambda x: x.ewm(span=5).mean().shift())
df['ava_xg'] = df.groupby('away_team_name')['team_b_xg'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avah_xg'] = df.groupby('home_team_name')['team_b_xg'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avaa_xg'] = df.groupby('away_team_name')['team_a_xg'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avposh'] = df.groupby('home_team_name')['home_team_possession'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avposa'] = df.groupby('away_team_name')['away_team_possession'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avshoth'] = df.groupby('home_team_name')['home_team_shots_off_target'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avshota'] = df.groupby('away_team_name')['away_team_shots_off_target'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avshotth'] = df.groupby('home_team_name')['home_team_shots_on_target'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avshotta'] = df.groupby('away_team_name')['away_team_shots_on_target'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avfh'] = df.groupby('home_team_name')['home_team_fouls'].apply(lambda x: x.ewm(span=5).mean().shift())
df['avfa'] = df.groupby('away_team_name')['away_team_fouls'].apply(lambda x: x.ewm(span=5).mean().shift())
def btts_prob(a, b):   
	home_goals_vector = poisson(a).pmf(np.arange(0, 26))
	away_goals_vector = poisson(b).pmf(np.arange(0, 26))
	m = np.outer(home_goals_vector, away_goals_vector)
	s = np.sum(m[1:, 1:])
	return s
def over(a, b):   
	home_goals_vector = poisson(a).pmf(np.arange(0, 26))
	away_goals_vector = poisson(b).pmf(np.arange(0, 26))
	m = np.outer(home_goals_vector, away_goals_vector)
	q = np.sum(m[0, 2:]) + np.sum(m[1, 1:]) + np.sum(m[2:])
	return q
df['bttsprob'] = df.apply(lambda x: btts_prob(x.avh_xg, x.ava_xg), axis=1)
df['btsn'] = 1 - df.bttsprob 
df['over15'] = df.apply(lambda x: over(x.avh_xg, x.ava_xg), axis=1)
df['under15'] = 1 - df.over15

df['BTTS'] = np.where((df.home_team_goal_count > 0) & (df.away_team_goal_count > 0), 1, 0)
#df['Homewin'] = np.where((df.FTHG > df.FTAG), 1, 0)
df['Over'] = np.where((df.home_team_goal_count + df.away_team_goal_count) > 1, 1, 0)
df['under'] = np.where((df.home_team_goal_count + df.away_team_goal_count) < 4, 1, 0)
#df = df.sort_values('Date')
df.dropna(inplace=True)

x = df[['home_team_name', 'away_team_name', 'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
'average_goals_per_match_pre_match', 'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
'over_35_percentage_pre_match', 'over_45_percentage_pre_match', 'over_05_2HG_percentage_pre_match', 'over_15_2HG_percentage_pre_match',
'average_corners_per_match_pre_match','average_cards_per_match_pre_match',
'ftsh', 'ftsa', 'btsh', 'btsa', 'avgh', 'avga', 'avghht', 'avgaht', 'avcgh', 'avcga', 'avh_xg', 'ava_xg', 'avah_xg',
'avaa_xg', 'avposh', 'avposa', 'avshoth', 'avshota', 'avshotth', 'avshotta', 'bttsprob', 'btsn',
'avfh', 'avfa', 'odds_btts_yes', 'odds_btts_no', 'odds_ft_home_team_win' ,'odds_ft_draw', 'odds_ft_away_team_win','over15', 'under15']]
y = df['total_goal_count']
#print(y.value_counts(normalize=True))
#print(df.to_string())
model1 = RandomForestRegressor(n_jobs=-1)
#model = GradientBoostingClassifier(n_estimators=500)
#model = MLPClassifier(max_iter=1000, shuffle=False)
#model = VotingClassifier([('clf1', model1), ('clf2', model2), ('clf3', model3)], voting='soft')
sel1 = SelectFromModel(model1)
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer(
    [('OHR', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [0,1])],
    remainder = 'passthrough'
    )

pipe = make_pipeline(ohe, sel1, model1)
pipe.fit(x,y)
#print(pipe.score(x, y))
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print("MSE:", mean_squared_error(y_test, y_pred))
df['pred_tot'] = pipe.predict(x)
#print(df.to_string())
x = df[['home_team_name', 'away_team_name', 'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
'average_goals_per_match_pre_match', 'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
'over_35_percentage_pre_match', 'over_45_percentage_pre_match', 'over_05_2HG_percentage_pre_match', 'over_15_2HG_percentage_pre_match',
'average_corners_per_match_pre_match','average_cards_per_match_pre_match',
'ftsh', 'ftsa', 'btsh', 'btsa', 'avgh', 'avga', 'avghht', 'avgaht', 'avcgh', 'avcga', 'avh_xg', 'ava_xg', 'avah_xg',
'avaa_xg', 'avposh', 'avposa', 'avshoth', 'avshota', 'avshotth', 'avshotta', 'bttsprob', 'btsn',
'avfh', 'avfa', 'odds_btts_yes', 'odds_btts_no', 'odds_ft_home_team_win' ,'odds_ft_draw', 'odds_ft_away_team_win','over15', 'under15']]
y = df['home_team_goal_count']
#print(y.value_counts(normalize=True))
#print(df.to_string())
model1 = RandomForestRegressor(n_jobs=-1)
#model = GradientBoostingClassifier(n_estimators=500)
#model = MLPClassifier(max_iter=1000, shuffle=False)
#model = VotingClassifier([('clf1', model1), ('clf2', model2), ('clf3', model3)], voting='soft')
sel1 = SelectFromModel(model1)
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer(
    [('OHR', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [0,1])],
    remainder = 'passthrough'
    )

pipe = make_pipeline(ohe, sel1, model1)
pipe.fit(x,y)
#print(pipe.score(x, y))
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print("MSE:", mean_squared_error(y_test, y_pred))
df['pred_home'] = pipe.predict(x)
x = df[['home_team_name', 'away_team_name', 'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
'average_goals_per_match_pre_match', 'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
'over_35_percentage_pre_match', 'over_45_percentage_pre_match', 'over_05_2HG_percentage_pre_match', 'over_15_2HG_percentage_pre_match',
'average_corners_per_match_pre_match','average_cards_per_match_pre_match',
'ftsh', 'ftsa', 'btsh', 'btsa', 'avgh', 'avga', 'avghht', 'avgaht', 'avcgh', 'avcga', 'avh_xg', 'ava_xg', 'avah_xg',
'avaa_xg', 'avposh', 'avposa', 'avshoth', 'avshota', 'avshotth', 'avshotta', 'bttsprob', 'btsn',
'avfh', 'avfa', 'odds_btts_yes', 'odds_btts_no', 'odds_ft_home_team_win' ,'odds_ft_draw', 'odds_ft_away_team_win','over15', 'under15']]
y = df['away_team_goal_count']
#print(y.value_counts(normalize=True))
#print(df.to_string())
model1 = RandomForestRegressor(n_jobs=-1)
#model = GradientBoostingClassifier(n_estimators=500)
#model = MLPClassifier(max_iter=1000, shuffle=False)
#model = VotingClassifier([('clf1', model1), ('clf2', model2), ('clf3', model3)], voting='soft')
sel1 = SelectFromModel(model1)
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer(
    [('OHR', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [0,1])],
    remainder = 'passthrough'
    )

pipe = make_pipeline(ohe, sel1, model1)
pipe.fit(x,y)
#print(pipe.score(x, y))
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print("MSE:", mean_squared_error(y_test, y_pred))
df['pred_away'] = pipe.predict(x)
x = df[['home_team_name', 'away_team_name', 'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
'average_goals_per_match_pre_match', 'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
'over_35_percentage_pre_match', 'over_45_percentage_pre_match', 'over_05_2HG_percentage_pre_match', 'over_15_2HG_percentage_pre_match',
'average_corners_per_match_pre_match','average_cards_per_match_pre_match',
'ftsh', 'ftsa', 'btsh', 'btsa', 'avgh', 'avga', 'avghht', 'avgaht', 'avcgh', 'avcga', 'avh_xg', 'ava_xg', 'avah_xg',
'avaa_xg', 'avposh', 'avposa', 'avshoth', 'avshota', 'avshotth', 'avshotta', 'bttsprob', 'btsn', 'pred_tot', 'pred_home', 'pred_away',
'avfh', 'avfa', 'odds_btts_yes', 'odds_btts_no', 'odds_ft_home_team_win' ,'odds_ft_draw', 'odds_ft_away_team_win','over15', 'under15']]
y = df['Over']
print(y.value_counts(normalize=True))
#print(df.to_string())
#model = LogisticRegression(C=1.1, penalty='l2', max_iter=3000)
#model = RandomForestClassifier(n_estimators=2000, class_weight='balanced', n_jobs=-1)
model = GradientBoostingClassifier(n_estimators=100)
#model = MLPClassifier(max_iter=1000, shuffle=False)
#model = VotingClassifier([('clf1', model1), ('clf2', model2), ('clf3', model3)], voting='soft')
sel = SelectFromModel(model)
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer(
    [('OHR', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [0,1])],
    remainder = 'passthrough'
    )
#scaler = StandardScaler()
#rfe = RFECV(estimator=LogisticRegression(max_iter=3000), n_jobs=-1)
pipe = make_pipeline(ohe, sel, model)
#p = cross_val_score(pipe, x, y, scoring='roc_auc', cv=ts, n_jobs=-1, error_score='raise')
#print(np.mean(p))
pipe.fit(x,y)
#print(pipe.score(x, y))
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False)
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
y_pred = pipe.predict(x_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("f1 score:", f1_score(y_test, y_pred))
print("auc score:", roc_auc_score(y_test, y_pred))
y_pred_new_threshold = (pipe.predict_proba(x_test)[:,1]>=0.65).astype(int)
print("Accuracy Threshold of 60%:", round(accuracy_score(y_test, y_pred_new_threshold), 3))
print("Precision Threshold of 60%:", round(precision_score(y_test, y_pred_new_threshold), 3))
print("Recall Threshold of 60%:", round(recall_score(y_test, y_pred_new_threshold), 3))
print("Auc score Threshold of 60%:", round(roc_auc_score(y_test, y_pred_new_threshold), 3))
df['pred'] = pipe.predict(x)
df['prob'] = pipe.predict_proba(x)[:,1]
print(df.to_string())