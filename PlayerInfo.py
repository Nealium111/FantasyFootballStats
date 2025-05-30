import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import math

st.set_page_config(layout="wide")

# Year selection
years = st.multiselect("Select Season(s)", list(range(2015, 2025)), default=[2024])

@st.cache_data
def load_pbp(years):
    return nfl.import_pbp_data(years)

if years:
    pbp = load_pbp(years)
else:
    st.warning("Please select at least one year.")
    st.stop()

# Load players data
@st.cache_data
def load_players():
    return nfl.import_players()

players = load_players()

offensive_positions = ['QB', 'RB', 'WR', 'TE']
offensive_players = players[players['position'].isin(offensive_positions)]
player_names = sorted(offensive_players['display_name'].unique())

st.title("Fantasy Football Player Comparison Tool")

col1, col2 = st.columns(2)
with col1:
    selected_players = st.multiselect("Choose Players", player_names)

fantasy_stats = [
    'targets',
    'receptions',
    'receiving_yards',
    'rushing_yards',
    'receiving_touchdowns',
    'rushing_touchdowns',
    'fumbles_lost',
    'interceptions_thrown',
    'passing_yards',
    'passing_touchdowns',
    'two_point_conversion',
    'yards_after_catch',
    'fantasy_points'
]

with col2:
    selected_stats = st.multiselect("Choose Stats", fantasy_stats)

def get_player_id(name):
    matches = offensive_players[offensive_players['display_name'] == name]
    if matches.empty:
        return None
    return matches.iloc[0]['gsis_id']

if selected_players and selected_stats:
    n_stats = len(selected_stats)
    cols = 2
    rows = math.ceil(n_stats / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
    axs = axs.flatten()

    totals = []

    # Create full index for (season, week)
    all_weeks = pd.MultiIndex.from_product([
        sorted(years), list(range(1, 19))
    ], names=["season", "week"])

    for i, stat in enumerate(selected_stats):
        ax = axs[i]

        for player in selected_players:
            player_id = get_player_id(player)
            if not player_id:
                continue

            if stat in ['targets', 'receptions', 'receiving_touchdowns']:
                if stat == 'targets':
                    df_stat = pbp[(pbp['receiver_player_id'] == player_id) & (pbp['pass_attempt'] == 1)]
                    grouped = df_stat.groupby(['season', 'week']).size()
                elif stat == 'receptions':
                    df_stat = pbp[(pbp['receiver_player_id'] == player_id) & (pbp['complete_pass'] == 1)]
                    grouped = df_stat.groupby(['season', 'week']).size()
                elif stat == 'receiving_touchdowns':
                    df_stat = pbp[(pbp['receiver_player_id'] == player_id) & (pbp['complete_pass'] == 1) & (pbp['touchdown'] == 1)]
                    grouped = df_stat.groupby(['season', 'week']).size()
            elif stat in ['receiving_yards', 'yards_after_catch']:
                df_stat = pbp[pbp['receiver_player_id'] == player_id]
                grouped = df_stat.groupby(['season', 'week'])[stat].sum()
            elif stat in ['rushing_yards', 'rushing_touchdowns', 'fumbles_lost']:
                df = pbp[pbp['rusher_player_id'] == player_id].copy()
                if stat == 'rushing_touchdowns':
                    df_stat = df[df['touchdown'] == 1]
                    grouped = df_stat.groupby(['season', 'week']).size()
                elif stat == 'fumbles_lost':
                    if 'fumbles_lost' not in df.columns:
                        df['fumbles_lost'] = ((df['fumble'] == 1) & (df['fumble_lost'] == 1)).astype(int)
                    grouped = df.groupby(['season', 'week'])['fumbles_lost'].sum()
                else:
                    grouped = df.groupby(['season', 'week'])[stat].sum()
            elif stat in ['interceptions_thrown', 'passing_yards', 'passing_touchdowns', 'two_point_conversion']:
                df = pbp[pbp['passer_player_id'] == player_id]
                grouped = df.groupby(['season', 'week'])[stat].sum()
            elif stat == 'fantasy_points':
                df_stat = pbp[(pbp['receiver_player_id'] == player_id) |
                              (pbp['rusher_player_id'] == player_id) |
                              (pbp['passer_player_id'] == player_id)].copy()

                player_row = players[players['gsis_id'] == player_id]
                is_te = not player_row.empty and player_row.iloc[0]['position'] == 'TE'

                df_stat['fantasy_points'] = 0

                receiving_tds = ((df_stat['receiver_player_id'] == player_id) &
                                 (df_stat['complete_pass'] == 1) &
                                 (df_stat['touchdown'] == 1)).astype(int)
                receiving_2pt = ((df_stat['two_point_conv_result'] == 'success') &
                                 (df_stat['receiver_player_id'] == player_id)).astype(int)
                df_stat.loc[df_stat['receiver_player_id'] == player_id, 'fantasy_points'] += (
                    df_stat['receiving_yards'] * 0.1 +
                    receiving_tds * 6 +
                    receiving_2pt * 2 +
                    (1.0 if is_te else 0.5) * df_stat['complete_pass']
                )

                rushing_tds = ((df_stat['rusher_player_id'] == player_id) & (df_stat['touchdown'] == 1)).astype(int)
                rushing_2pt = ((df_stat['two_point_conv_result'] == 'success') & (df_stat['rusher_player_id'] == player_id)).astype(int)
                df_stat.loc[df_stat['rusher_player_id'] == player_id, 'fantasy_points'] += (
                    df_stat['rushing_yards'] * 0.1 +
                    rushing_tds * 6 +
                    rushing_2pt * 2
                )

                passing_tds = ((df_stat['passer_player_id'] == player_id) &
                               (df_stat['complete_pass'] == 1) &
                               (df_stat['touchdown'] == 1)).astype(int)
                passing_2pt = ((df_stat['two_point_conv_result'] == 'success') &
                               (df_stat['passer_player_id'] == player_id)).astype(int)
                df_stat.loc[df_stat['passer_player_id'] == player_id, 'fantasy_points'] += (
                    df_stat['passing_yards'] * 0.04 +
                    passing_tds * 4 +
                    passing_2pt * 2
                )

                df_stat['fumble_recovery_td'] = (
                    (df_stat['touchdown'] == 1) &
                    ((df_stat['fumble_recovery_1_player_id'] == player_id) |
                     (df_stat['fumble_recovery_2_player_id'] == player_id))
                ).astype(int)
                df_stat.loc[:, 'fantasy_points'] += df_stat['fumble_recovery_td'] * 6

                grouped = df_stat.groupby(['season', 'week'])['fantasy_points'].sum()
            else:
                continue

            # Align data to all season-week pairs
            grouped = grouped.reindex(all_weeks, fill_value=0)
            grouped = grouped.sort_index()

            x = list(range(len(grouped)))
            x_labels = [f"{season} W{week}" for season, week in grouped.index]

            ax.plot(x, grouped.values, marker='o', label=player)

            step = max(1, len(x_labels) // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(x_labels[::step], rotation=45, ha='right', fontsize=8)

            totals.append({
                "Player": player,
                "Stat": stat,
                "Total": grouped.sum()
            })

        ax.legend()
        ax.set_title(stat.replace('_', ' ').title())
        ax.set_xlabel('Season Week')
        ax.set_ylabel(stat.replace('_', ' ').title())
        ax.grid(True)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()

    season_df = pd.DataFrame(totals)
    season_df = season_df.pivot(index="Player", columns="Stat", values="Total").fillna(0)

    st.dataframe(season_df.style.format("{:.1f}"), use_container_width=True)
    st.pyplot(fig)

else:
    st.write("Please select at least one player and one stat to compare.")
