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

@st.cache_data
def load_rosters():
    return nfl.import_seasonal_rosters([2024])  # or include more years if needed

rosters = load_rosters()
players = load_players()



offensive_positions = ['QB', 'RB', 'WR', 'TE']
offensive_rosters = rosters[rosters['position'].isin(offensive_positions)]
offensive_players = players[players['position'].isin(offensive_positions)]

all_offensive_players = offensive_players['display_name'].unique()
roster_player_names = offensive_rosters['player_name'].unique()

# Rookies are in players but not in roster_player_names
rookies = [name for name in all_offensive_players if name not in roster_player_names]

# Combine and sort
player_names = sorted(set(list(roster_player_names) + rookies))

#selected_players = st.multiselect("Choose Players", player_names, key="player_select")

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
    # Try roster (has player_id)
    matches = offensive_rosters[offensive_rosters['player_name'] == name]
    if not matches.empty:
        if 'player_id' in matches.columns:
            return matches.iloc[0]['player_id']

    # Try players table
    matches = offensive_players[offensive_players['display_name'] == name]
    if not matches.empty:
        # Check for common ID fields
        for id_col in ['player_id', 'gsis_id', 'nfl_id', 'pfr_id']:
            if id_col in matches.columns and pd.notna(matches.iloc[0][id_col]):
                return matches.iloc[0][id_col]

    # Assign synthetic ID for rookie with no ID
    return f"rookie_{name.replace(' ', '_').lower()}"

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

import datetime

st.header("Dynasty Trade Calculator")

# Draft pick values (example scale, adjust as needed)
draft_pick_values = {
    'Draft Pick: 1st Round': 200,
    'Draft Pick: 2nd Round': 100,
    'Draft Pick: 3rd Round': 50,
    'Draft Pick: 4th Round': 25,
}

# Combined player + picks list for dropdown
combined_options = [""] + player_names + list(draft_pick_values.keys())

def calculate_player_rating_with_details(player_id, pbp, players, years):
    df_stat = pbp[
        (pbp['receiver_player_id'] == player_id) |
        (pbp['rusher_player_id'] == player_id) |
        (pbp['passer_player_id'] == player_id)
    ].copy()

    if df_stat.empty:
        total_value = 0
    else:
        # Calculate positive stats excluding touchdowns
        receiving_yds = df_stat.loc[df_stat['receiver_player_id'] == player_id, 'receiving_yards'].sum()
        rushing_yds = df_stat.loc[df_stat['rusher_player_id'] == player_id, 'rushing_yards'].sum()
        passing_yds = df_stat.loc[df_stat['passer_player_id'] == player_id, 'passing_yards'].sum()

        receptions = df_stat.loc[(df_stat['receiver_player_id'] == player_id) & (df_stat['complete_pass'] == 1), :].shape[0]
        targets = df_stat.loc[(df_stat['receiver_player_id'] == player_id) & (df_stat['pass_attempt'] == 1), :].shape[0]

        # Yards after catch (YAC)
        yac = df_stat.loc[df_stat['receiver_player_id'] == player_id, 'yards_after_catch'].sum()

        # Fumbles lost (negative stat - exclude or subtract)
        fumbles_lost = df_stat.loc[df_stat['rusher_player_id'] == player_id, 'fumbles_lost'].sum() if 'fumbles_lost' in df_stat.columns else 0

        # Interceptions thrown (negative, exclude)
        interceptions_thrown = df_stat.loc[df_stat['passer_player_id'] == player_id, 'interceptions_thrown'].sum() if 'interceptions_thrown' in df_stat.columns else 0

        # Combine positive stats with weighting (customize weights as needed)
        total_value = (
            receiving_yds * 0.1 +
            rushing_yds * 0.1 +
            passing_yds * 0.04 +
            receptions * 1 +
            targets * 0.5 +
            yac * 0.05
        )
        # You can optionally subtract negative stats if you want:
        # total_value -= (fumbles_lost * 2 + interceptions_thrown * 3)

    # Age and age factor
    player_row = players[players['gsis_id'] == player_id]
    if player_row.empty or pd.isna(player_row.iloc[0]['birth_date']):
        age_factor = 1.0
        age = None
    else:
        birth_date = pd.to_datetime(player_row.iloc[0]['birth_date'])
        age = (datetime.datetime.now() - birth_date).days / 365.25
        age_factor = max(0, min(1, (35 - age) / (35 - 21)))

    # Rookie baseline for no value players
    rookie_baseline_value = 100

    if total_value == 0:
        rating = rookie_baseline_value * age_factor
    else:
        rating = total_value * age_factor

    return rating, total_value, age_factor, age


def trade_side_ui(side_label, key_prefix):
    st.subheader(f"Trade Side {side_label}")

    # Init session state lists for slots if missing
    if f"{key_prefix}_slots" not in st.session_state:
        st.session_state[f"{key_prefix}_slots"] = [""]
    
    slots = st.session_state[f"{key_prefix}_slots"]

    # Add / Remove buttons
    cols = st.columns([1, 1])
    with cols[0]:
        if st.button(f"Add Slot ({side_label})", key=f"add_slot_{key_prefix}"):
            slots.append("")
    with cols[1]:
        if len(slots) > 1:
            if st.button(f"Remove Last Slot ({side_label})", key=f"remove_slot_{key_prefix}"):
                slots.pop()

    # Render dropdowns and update session_state
    new_slots = []
    for i, val in enumerate(slots):
        new_val = st.selectbox(f"Select Player or Pick #{i+1} ({side_label})", combined_options, index=combined_options.index(val) if val in combined_options else 0, key=f"{key_prefix}_slot_{i}")
        new_slots.append(new_val)

    st.session_state[f"{key_prefix}_slots"] = new_slots
    return new_slots

# === Show UI and compute ===

trade_a = trade_side_ui("A", "side_a")
trade_b = trade_side_ui("B", "side_b")

def compute_trade_value_detailed(slots):
    total_value = 0
    player_details = []
    for item in slots:
        if item.startswith("Draft Pick"):
            value = draft_pick_values.get(item, 0)
            total_value += value
            player_details.append({
                "Player": item,
                "Rating": value,
                "Fantasy Points": None,
                "Age Factor": None,
                "Age": None
            })
        elif item != "":
            pid = get_player_id(item)
            if pid:
                rating, total_fp, age_factor, age = calculate_player_rating_with_details(pid, pbp, players, years)
                total_value += rating
                player_details.append({
                    "Player": item,
                    "Rating": rating,
                    "Fantasy Points": total_fp,
                    "Age Factor": age_factor,
                    "Age": age
                })
    return total_value, player_details


if st.button("Calculate Trade Values"):

    value_a, details_a = compute_trade_value_detailed(trade_a)
    value_b, details_b = compute_trade_value_detailed(trade_b)

    st.write(f"**Trade Side A Value:** {value_a:.1f}")
    st.write(f"**Trade Side B Value:** {value_b:.1f}")


    if not any(trade_b):  # Only recommend if Side B is empty
        st.subheader("💡 Suggested Trade Matches for Side B")

        all_candidates = [p for p in combined_options if p not in trade_a and p != ""]

        candidate_values = []
        for candidate in all_candidates:
            if candidate.startswith("Draft Pick"):
                val = draft_pick_values.get(candidate, 0)
            else:
                pid = get_player_id(candidate)
                if not pid or str(pid).startswith("rookie_"):  # Skip rookies
                    continue
                # Confirm player is on 2024 roster
                if candidate not in offensive_rosters['player_name'].values:
                    continue
                val, _, _, _ = calculate_player_rating_with_details(pid, pbp, players, years)

            candidate_values.append((candidate, val))

        candidate_values = sorted(candidate_values, key=lambda x: abs(x[1] - value_a))

        # Show top 5 closest matches
        top_matches = candidate_values[:5]

        for name, val in top_matches:
            st.write(f"- **{name}**: Estimated Value = {val:.1f} (Diff = {abs(val - value_a):.1f})")

    st.markdown("---")
    st.subheader("Details for Trade")

    def safe_format(x, fmt):
        if x is None:
            return ""
        return fmt.format(x)

    df_a = pd.DataFrame(details_a)
    df_b = pd.DataFrame(details_b)

    st.dataframe(
        df_a.style.format({
            "Rating": lambda x: safe_format(x, "{:.1f}"),
            "Fantasy Points": lambda x: safe_format(x, "{:.1f}"),
            "Age Factor": lambda x: safe_format(x, "{:.2f}"),
            "Age": lambda x: safe_format(x, "{:.1f}")
        }),
        use_container_width=True
    )

    st.dataframe(
        df_b.style.format({
            "Rating": lambda x: safe_format(x, "{:.1f}"),
            "Fantasy Points": lambda x: safe_format(x, "{:.1f}"),
            "Age Factor": lambda x: safe_format(x, "{:.2f}"),
            "Age": lambda x: safe_format(x, "{:.1f}")
        }),
        use_container_width=True
    )
