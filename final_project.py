'''

Randall Candaso
ISTA 350
Final Project

The program takes data from previous NFL seasons since 1970 and creates three visual figures to represent
different statistical comparisons. A linear model is created and fitted to one of the visuals' data.

'''
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


def main():
    teams = pd.read_csv('Team Stats.csv', index_col=0)
    winners = pd.read_csv('SB Winners.csv')

    teams['Won'] = 0

    for i in winners.index:
        teams.loc[(teams['Season'] == winners.loc[i, 'Season']) & (teams['Team'] == winners.loc[i, 'Winner']), 'Won'] = 1
    df = teams.copy()

    make_fig_3(df)
    make_fig_2(df)
    make_fig_1(df)
    plt.show()


def get_ols_parameters(x, y):
    """

    x: independent variable
    y: independent variable

    This function calculates a OLS model and returns the slope, intercept, r^2, and p_val
    values from the model.

    return: linear parameters and statistics (list)

    """
    const = sm.add_constant(x)
    model = sm.OLS(y, const)
    results = model.fit()
    slope = results.params.iloc[1]
    inter = results.params.iloc[0]
    r_sq = results.rsquared
    p_val = results.pvalues.iloc[1]
    return [slope, inter, r_sq, p_val]


def make_fig_1(df):
    '''
    
    df: dataframe to be modeled (pd.DataFrame)

    A visual is created to represent the relationship of offensive data to win percentage. Additionally, a linear model is fitted 
    to the data and plotted to show the behavior of the relationship.
    
    '''
    metrics = ["Pass.Yds", "Pass.TD", "Rush.Yds", "Rush.TD"]
    titles = ["Pass Yards vs W-L%", "Pass TDs vs W-L%", "Rush Yards vs W-L%", "Rush TDs vs W-L%"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        data_won_true = df[df["Won"] == 1]
        data_won_false = df[df["Won"] == 0]

        ax.scatter(data_won_false[metric], data_won_false["W-L%"], color="green", label="SB: False", s=30,
                    alpha=0.7)
        ax.scatter(data_won_true[metric], data_won_true["W-L%"], color="yellow", label="SB: True", s=30,
                    alpha=0.7)
        
        params = get_ols_parameters(df.loc[:, metric], df.loc[:, "W-L%"])
        line = params[0] * df.loc[:, metric] + params[1]
        ax.plot(df.loc[:, metric], line, color="blue", linestyle="dashed", linewidth=2, label="Linear Fit")

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel(metrics[i], fontsize=10)
        ax.set_ylabel("Win Percentage", fontsize=10)
        ax.set_ylim(0, None)

    axes[0].legend(title="Super Bowl Winners", loc="best")
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Passing Stats to Win Percentage", fontsize=16, y=0.95)


def make_fig_2(df):
    '''
    
    df: dataframe to be visualized (pd.DataFrame)

    The season passing and rushing yards are plotted on a barchart for each playoff eligible team of the 2024 NFL season. 
    It should be noted that the bars are colored the same for every team except the 2024 Super Bowl winning Eagles. 
    
    '''
    new = df[df['Season'] == 2024]
    x = np.arange(len(new.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, new['Pass.Yds'], width, label='Pass Yards', color='black')
    bars2 = ax.bar(x + width/2, new['Rush.Yds'], width, label='Rush Yards', color='grey')

    for i, team in enumerate(new['Team']):
        if team == 'PHI':
            bars1[i].set_color('green')
            bars2[i].set_color('lightgreen')

    ax.set_ylabel('Total Yards')
    ax.set_xlabel('Team')
    ax.set_title('Yards by 2024 Playoff-Eligible Teams')
    ax.set_xticks(x)
    ax.set_xticklabels(new['Team'])
    ax.legend()

    plt.tight_layout()


def make_fig_3(df):
    '''
    
    df: dataframe to be visualized (pd.DataFrame)

    The total defensive stats for all teams for each year is plotted on a line plot. The stats plotted include total 
    defensive plays, total passing yards allowed, and total rushing yards allowed. 
    
    '''
    d_plays = df.groupby('Season')['DPly'].mean()
    opp_pass_yds = df.groupby('Season')['Opp.Pass.Yds'].mean()
    opp_rush_yds = df.groupby('Season')['Opp.Rush.Yds'].mean()

    plt.figure(figsize=(14, 8))

    plt.plot(d_plays.index, d_plays, color='blue', linestyle='--', label='Defensive Plays')
    plt.plot(opp_pass_yds.index, opp_pass_yds, color='green', linestyle='--', label='Opponent Passing Yards')
    plt.plot(opp_rush_yds.index, opp_rush_yds, color='red', linestyle='--', label='Opponent Rushing Yards')

    plt.xlim(1970, 2025)
    plt.xticks(range(1970, 2025, 5), rotation=45)
    plt.xlabel('Season')
    plt.ylabel('Average #')
    plt.legend(title='Defensive Stats')
    plt.title('Defensive Stats Over Time')


main()