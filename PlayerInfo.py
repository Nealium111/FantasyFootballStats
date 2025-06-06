import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
from itertools import combinations, product

st.set_page_config(layout="wide")



@st.cache_data
def load_pbp(years):
    df = nfl.import_pbp_data(years)

    # Keep only rows relevant to fantasy scoring
    df = df[df['play_type'].isin(['pass', 'run', 'qb_spike', 'qb_kneel', 'sack'])]

    # Reduce columns to only what's used
    cols_to_keep = [
        'season', 'week',
        'receiver_player_id', 'rusher_player_id', 'passer_player_id',
        'complete_pass', 'pass_attempt', 'passing_yards', 'passing_touchdowns',
        'receiving_yards', 'rushing_yards', 'receiving_touchdowns', 'rushing_touchdowns',
        'two_point_conv_result', 'yards_after_catch',
        'interceptions_thrown', 'fumble', 'fumble_lost',
        'fumble_recovery_1_player_id', 'fumble_recovery_2_player_id',
        'touchdown'
    ]

    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols].copy()

    return df

# Load players data
@st.cache_data
def load_players():
    return nfl.import_players()

@st.cache_data
def load_rosters():
    return nfl.import_seasonal_rosters([2024])  # or include more years if needed

@st.cache_data
def get_name_id_map(rosters, players):
    name_id = {}

    for _, row in rosters.iterrows():
        name = row['player_name']
        pid = row.get('player_id')
        if pd.notna(name) and pd.notna(pid):
            name_id[name] = pid

    for _, row in players.iterrows():
        name = row['display_name']
        pid = row.get('gsis_id')
        if name not in name_id and pd.notna(pid):
            name_id[name] = pid

    return name_id

rosters = load_rosters()
players = load_players()

name_id_map = get_name_id_map(rosters, players)

# Year selection
years = st.multiselect("Select Season(s)", list(range(2015, 2025)), default=[2024])

if years:
    pbp = load_pbp(years)
else:
    st.warning("Please select at least one year.")
    st.stop()

def get_player_id(name):
    return name_id_map.get(name, f"rookie_{name.replace(' ', '_').lower()}")

def calculate_player_rating_with_details(player_id, pbp, players, years, receiving_yds_weight, rushing_yds_weight, passing_yds_weight, receptions_weight, targets_weight, yac_weight, rec_tds_weight, rush_tds_weight, pass_tds_weight, age_weight):
    df_stat = pbp[
        (pbp['receiver_player_id'] == player_id) |
        (pbp['rusher_player_id'] == player_id) |
        (pbp['passer_player_id'] == player_id)
    ].copy()

    if df_stat.empty:
        total_value = 0
    else:
        # Positive stats
        receiving_yds = df_stat.loc[df_stat['receiver_player_id'] == player_id, 'receiving_yards'].sum()
        rushing_yds = df_stat.loc[df_stat['rusher_player_id'] == player_id, 'rushing_yards'].sum()
        passing_yds = df_stat.loc[df_stat['passer_player_id'] == player_id, 'passing_yards'].sum()

        receptions = df_stat.loc[(df_stat['receiver_player_id'] == player_id) & (df_stat['complete_pass'] == 1)].shape[0]
        targets = df_stat.loc[(df_stat['receiver_player_id'] == player_id) & (df_stat['pass_attempt'] == 1)].shape[0]
        yac = df_stat.loc[df_stat['receiver_player_id'] == player_id, 'yards_after_catch'].sum()

        # Touchdowns
        rec_tds = df_stat.loc[
            (df_stat['receiver_player_id'] == player_id) &
            (df_stat['complete_pass'] == 1) &
            (df_stat['touchdown'] == 1)
        ].shape[0]

        rush_tds = df_stat.loc[
            (df_stat['rusher_player_id'] == player_id) &
            (df_stat['touchdown'] == 1)
        ].shape[0]

        pass_tds = df_stat.loc[
            (df_stat['passer_player_id'] == player_id) &
            (df_stat['complete_pass'] == 1) &
            (df_stat['touchdown'] == 1)
        ].shape[0]

        # Total value based on sliders
        total_value = (
            receiving_yds * receiving_yds_weight +
            rushing_yds * rushing_yds_weight +
            passing_yds * passing_yds_weight +
            receptions * receptions_weight +
            targets * targets_weight +
            yac * yac_weight +
            rec_tds * rec_tds_weight +
            rush_tds * rush_tds_weight +
            pass_tds * pass_tds_weight
        )

    # Age and age factor
    player_row = players[players['gsis_id'] == player_id]
    if player_row.empty or pd.isna(player_row.iloc[0]['birth_date']):
        age_factor = 1.0
        age = None
    else:
        birth_date = pd.to_datetime(player_row.iloc[0]['birth_date'])
        age = (datetime.datetime.now() - birth_date).days / 365.25

        if age <= 21:
            age_factor = 8.0
        elif age <= 28:
            # Slow decline from 8.0 at 21 to 7.0 at 28
            age_factor = 8.0 - (age - 21) * (1.0 / 7)
        elif age <= 35:
            # Faster decline from 7.0 at 28 to 2.0 at 35
            age_factor = 7.0 - (age - 28) * (5.0 / 7)
        else:
            # Minimal value after 35
            age_factor = 1.0

        age_factor = max(0.5, age_factor)  # Clamp
    rookie_baseline_value = 100

    if total_value == 0:
        rating = rookie_baseline_value * (1 + (age_factor - 5.0) * age_weight * 0.1)
    else:
        rating = total_value * (1 + (age_factor - 5.0) * age_weight * 0.1)

    return rating, total_value, age_factor, age

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
                rating, total_fp, age_factor, age = calculate_player_rating_with_details(
                    pid, pbp, players, years,
                    receiving_yds_weight,
                    rushing_yds_weight,
                    passing_yds_weight,
                    receptions_weight,
                    targets_weight,
                    yac_weight,
                    rec_tds_weight,
                    rush_tds_weight,
                    pass_tds_weight,
                    age_weight
                )
                total_value += rating
                player_details.append({
                    "Player": item,
                    "Rating": rating,
                    "Fantasy Points": total_fp,
                    "Age Factor": age_factor,
                    "Age": age
                })
    return total_value, player_details


offensive_positions = ['QB', 'RB', 'WR', 'TE']
offensive_rosters = rosters[rosters['position'].isin(offensive_positions)]
offensive_players = players[players['position'].isin(offensive_positions)]

all_offensive_players = offensive_players['display_name'].unique()
roster_player_names = offensive_rosters['player_name'].unique()

# Rookies are in players but not in roster_player_names
rookies = list(set(all_offensive_players) - set(roster_player_names))

# Combine and sort
player_names = sorted(set(list(roster_player_names) + rookies))

tab1,tab2=st.tabs(["Player Comparison", "Dynasty Trade Calculator"])

with tab1:
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

with tab2:
    st.header("Dynasty Trade Calculator")
    st.sidebar.header("⚙️ Trade Settings")

    age_weight = st.sidebar.slider(
        "Age Influence Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Controls how much age impacts a player's trade value. 0 = No age impact, 1 = Full influence"
    )

    st.sidebar.subheader("📊 Stat Weight Settings")

    receiving_yds_weight = st.sidebar.slider("Receiving Yards Weight", 0.0, 0.2, 0.1, step=0.01)
    rushing_yds_weight = st.sidebar.slider("Rushing Yards Weight", 0.0, 0.2, 0.1, step=0.01)
    passing_yds_weight = st.sidebar.slider("Passing Yards Weight", 0.0, 0.1, 0.04, step=0.01)
    receptions_weight = st.sidebar.slider("Receptions Weight", 0.0, 2.0, 1.0, step=0.1)
    targets_weight = st.sidebar.slider("Targets Weight", 0.0, 1.0, 0.5, step=0.1)
    yac_weight = st.sidebar.slider("Yards After Catch (YAC) Weight", 0.0, 0.2, 0.05, step=0.01)

    # Touchdown sliders
    rec_tds_weight = st.sidebar.slider("Receiving TDs Weight", 0.0, 10.0, 6.0, step=0.5)
    rush_tds_weight = st.sidebar.slider("Rushing TDs Weight", 0.0, 10.0, 6.0, step=0.5)
    pass_tds_weight = st.sidebar.slider("Passing TDs Weight", 0.0, 10.0, 4.0, step=0.5)

    # Draft pick values (example scale, adjust as needed)
    draft_pick_values = {
        'Draft Pick: 1st Round': 200,
        'Draft Pick: 2nd Round': 100,
        'Draft Pick: 3rd Round': 50,
        'Draft Pick: 4th Round': 25,
    }

    # Combined player + picks list for dropdown
    combined_options = [""] + player_names + list(draft_pick_values.keys())


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

    recommendation_type = st.selectbox(
    "Choose recommendation type for Side B:",
    ["1 Player", "2 Players", "1 Player + 1 Draft Pick"]
    )
    if st.button("Calculate Trade Values"):

        value_a, details_a = compute_trade_value_detailed(trade_a)
        value_b, details_b = compute_trade_value_detailed(trade_b)

        st.write(f"**Trade Side A Value:** {value_a:.1f}")
        st.write(f"**Trade Side B Value:** {value_b:.1f}")
        with st.spinner("Loading recommendations..."):
            min_value_threshold = 100.0  # or whatever minimum rating you want
            if not any(trade_b):  # Only recommend if Side B is empty
                from itertools import combinations


                # Filter out rookies and Side A players
                valid_candidates = []
                for candidate in combined_options:
                    if candidate in trade_a or candidate == "":
                        continue
                    if candidate.startswith("Draft Pick"):
                        val = draft_pick_values.get(candidate, 0)
                        valid_candidates.append((candidate, val))
                    else:
                        pid = get_player_id(candidate)
                        if not pid or str(pid).startswith("rookie_"):
                            continue
                        if candidate not in offensive_rosters['player_name'].values:
                            continue
                        val, _, _, _ = calculate_player_rating_with_details(pid, pbp, players, years, receiving_yds_weight, rushing_yds_weight, passing_yds_weight, receptions_weight, targets_weight, yac_weight, rec_tds_weight, rush_tds_weight, pass_tds_weight, age_weight)
                        if val < min_value_threshold:
                            continue  # skip low-value players
                        valid_candidates.append((candidate, val))
    
            # Separate players and draft picks
            valid_players = []
            valid_picks = []

            for candidate, val in valid_candidates:
                if candidate.startswith("Draft Pick"):
                    valid_picks.append((candidate, val))
                else:
                    valid_players.append((candidate, val))

            # Add user choice for recommendation type before this block:
            # recommendation_type = st.selectbox(
            #     "Choose recommendation type for Side B:",
            #     ["1 Player", "2 Players", "1 Player + 1 Draft Pick"]
            # )

            all_combos = []

            if recommendation_type == "1 Player":
                # Just single players only
                for player in valid_players:
                    names = [player[0]]
                    total = player[1]
                    diff = abs(total - value_a)
                    all_combos.append((names, total, diff))

            elif recommendation_type == "2 Players":
                # Combinations of 2 players only
                for combo in combinations(valid_players, 2):
                    names = [x[0] for x in combo]
                    total = sum(x[1] for x in combo)
                    diff = abs(total - value_a)
                    all_combos.append((names, total, diff))

            elif recommendation_type == "1 Player + 1 Draft Pick":
                # All combinations of 1 player and 1 draft pick
                for player, pick in product(valid_players, valid_picks):
                    names = [player[0], pick[0]]
                    total = player[1] + pick[1]
                    diff = abs(total - value_a)
                    all_combos.append((names, total, diff))

            # Sort by difference to Side A
            all_combos = sorted(all_combos, key=lambda x: x[2])
            st.subheader("💡 Suggested Trade Combinations for Side B")
            st.markdown("**Top 5 Closest Value Matches:**")
            for names, total, diff in all_combos[:5]:
                st.write(f"- {' + '.join(names)}: Total Value = {total:.1f} (Diff = {diff:.1f})")

        st.markdown("---")
        st.subheader("Details for Trade")
        with st.expander("🔧 Current Stat Weights Used in Trade Valuation"):
            st.json({
                "Receiving Yards": receiving_yds_weight,
                "Rushing Yards": rushing_yds_weight,
                "Passing Yards": passing_yds_weight,
                "Receptions": receptions_weight,
                "Targets": targets_weight,
                "YAC": yac_weight,
                "Receiving TDs": rec_tds_weight,
                "Rushing TDs": rush_tds_weight,
                "Passing TDs": pass_tds_weight,
                "Age Weight": age_weight
            })

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
