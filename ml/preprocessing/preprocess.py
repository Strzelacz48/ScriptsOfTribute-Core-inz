"""
preprocess.py — Convert training_data.jsonl into numpy arrays ready for NN training.

Run:
    python preprocess.py --input ../../training_data.jsonl --output dataset.npz

Output arrays saved in dataset.npz:
    states   : float32 (N, STATE_DIM)   — one row per decision point
    actions  : int64   (N, 2)           — [command_type, card_or_patron_id]
    outcomes : float32 (N,)             — 1.0 = won, 0.0 = lost
"""

import argparse
import json
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants — must match the C# CardId and CommandEnum enums exactly
# ---------------------------------------------------------------------------

NUM_CARDS    = 125   # CardId values 0..124
NUM_COMMANDS = 7     # CommandEnum: PLAY_CARD=0 ACTIVATE_AGENT=1 ATTACK=2
                     #              BUY_CARD=3  CALL_PATRON=4    MAKE_CHOICE=5
                     #              END_TURN=6  (END_TURN records are never logged)
NUM_PATRONS  = 10    # PatronId values 0..9 (including TREASURY=0)

# Card zones we encode for the current player (each becomes a count vector of length NUM_CARDS)
MY_ZONES    = ["hand", "draw", "cool", "played", "known"]

# Card zones we encode for the enemy player
ENEMY_ZONES = ["en_hd", "en_cool", "en_played"]

# Scalar features pulled directly from each record
# Extend this list if you want to add derived features later
MY_SCALARS    = ["me_p", "me_c", "me_pw", "me_pc"]   # prestige, coins, power, patronCalls
ENEMY_SCALARS = ["en_p", "en_c", "en_pw"]             # prestige, coins, power

# Total state vector size — update this if you change the encoding below
# MY_ZONES * NUM_CARDS + ENEMY_ZONES * NUM_CARDS + agent slots + scalars + patrons
STATE_DIM = (
    len(MY_ZONES)    * NUM_CARDS +   # 5 * 125 = 625
    len(ENEMY_ZONES) * NUM_CARDS +   # 3 * 125 = 375
    NUM_CARDS +                      # my agent HP vector (one slot per card id)
    NUM_CARDS +                      # enemy agent HP vector
    NUM_CARDS +                      # tavern available (binary)
    NUM_CARDS +                      # tavern remaining (count)
    len(MY_SCALARS) +                # 4
    len(ENEMY_SCALARS) +             # 3
    NUM_PATRONS                      # patron states: -1 / 0 / 1 per patron
)   # = 1640


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    """Load all records from a JSONL file. Each line is one move decision."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------

# Convert a list of CardId integers into a count vector of length NUM_CARDS.
# Index i = how many copies of card i are in this zone.
def cards_to_count_vector(card_ids: list[int]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card_id in card_ids:
        vec[card_id] += 1
    return vec


# Like cards_to_count_vector but clips to 0/1 — useful for tavern available
# (you either can buy a card or you can't, not twice).
def cards_to_binary_vector(card_ids: list[int]) -> np.ndarray:
    vec = cards_to_count_vector(card_ids)
    np.clip(vec, 0, 1, out=vec)
    return vec


# Convert agent list [[cardId, hp], ...] to a vector of length NUM_CARDS
# where index i = current HP of the agent with that card id (0 if not on board).
def agents_to_hp_vector(agents: list[list[int]]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for agent in agents:
        card_id, hp = agent
        vec[card_id] = hp
    return vec


# Convert patron state list [{"id": patronId, "v": -1/0/1}, ...] to a
# fixed-length vector of length NUM_PATRONS.
# Patrons not in this game stay at 0 (neutral).
def patrons_to_vector(patron_list: list[dict]) -> np.ndarray:
    vec = np.zeros(NUM_PATRONS, dtype=np.float32)
    for patron in patron_list:
        vec[patron["id"]] = patron["v"]
    return vec


# ---------------------------------------------------------------------------
# Full state encoder
# ---------------------------------------------------------------------------

def encode_state(record: dict) -> np.ndarray:
    """
    Build the full flat feature vector for one decision point.
    Concatenates all encoded zones and scalars into a single float32 array
    of length STATE_DIM.

    The order must stay consistent between training and inference:
        [my_zones... | enemy_zones... | my_agents | en_agents |
         tv_available | tv_remaining | my_scalars | enemy_scalars | patrons]

    Hint: scalars can be extracted as np.array([record["me_p"], ...], dtype=np.float32)
    """
    parts = []

    # --- My card zones (5 count vectors) ---
    for zone in MY_ZONES:
        parts.append(cards_to_count_vector(record[zone]))

    # --- Enemy card zones (3 count vectors) ---
    for zone in ENEMY_ZONES:
        parts.append(cards_to_count_vector(record[zone]))

    # --- Agent HP vectors ---
    parts.append(agents_to_hp_vector(record["my_ag"]))
    parts.append(agents_to_hp_vector(record["en_ag"]))

    # --- Tavern ---
    parts.append(cards_to_binary_vector(record["tv_av"]))
    parts.append(cards_to_count_vector(record["tv_rm"]))

    # --- Scalar features ---
    parts.append(np.array([record[k] for k in MY_SCALARS], dtype=np.float32))
    parts.append(np.array([record[k] for k in ENEMY_SCALARS], dtype=np.float32))

    # --- Patron states ---
    parts.append(patrons_to_vector(record["pat"]))
    
    return np.concatenate(parts, dtype=np.float32)


def encode_action(record: dict) -> np.ndarray:
    """
    Encode the chosen action as a 2-element array: [command_type, target_id].
    target_id is the first element of act_ids, or -1 if act_ids is empty.

    This is used as the label for the imitation learning policy head.
    """
    command_type = record["act"]
    target_id = record["act_ids"][0] if record["act_ids"] else -1
    return np.array([command_type, target_id], dtype=np.int64)



# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all records into three parallel arrays:
        states   float32 (N, STATE_DIM)
        actions  int64   (N, 2)
        outcomes float32 (N,)   — record["won"] is already 0 or 1

    collect into lists, then np.stack into arrays.
    Skip (or assert) any record where won == -1 (shouldn't happen after GameEnd).
    """
    states, actions, outcomes = [], [], []

    for record in tqdm(records, desc="Encoding"):
        if record["won"] == -1:
            continue
        states.append(encode_state(record))
        actions.append(encode_action(record))
        outcomes.append(float(record["won"]))

    return np.stack(states), np.stack(actions), np.array(outcomes, dtype=np.float32)
    

# ---------------------------------------------------------------------------
# Card win rate analysis
# ---------------------------------------------------------------------------

def compute_card_win_rates(records: list[dict]) -> dict[int, float]:
    """
    For each CardId, compute the fraction of records where the card appeared
    in my hand/draw/cool/played AND the game was won.

    Returns dict: {card_id: win_rate}  (only cards that appeared at least once)

    Useful for sanity-checking the data and as an optional extra feature.

    """
    appearances: dict[int, int] = {}
    wins: dict[int, int] = {}

    # Exclude "known" — it's a subset of "draw", so including it would double-count those cards
    count_zones = [z for z in MY_ZONES if z != "known"]

    for record in records:
        if record["won"] == -1:
            continue
        for zone in count_zones:
            for card_id in record[zone]:
                appearances[card_id] = appearances.get(card_id, 0) + 1
                wins[card_id]        = wins.get(card_id, 0) + record["won"]

    return {card_id: wins[card_id] / appearances[card_id] for card_id in appearances}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess training_data.jsonl into numpy arrays.")
    parser.add_argument("--input",  required=True,             help="Path to training_data.jsonl")
    parser.add_argument("--output", default="dataset.npz",     help="Output .npz file path")
    parser.add_argument("--winrates", action="store_true",     help="Also print card win rates")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    records = load_records(args.input)
    print(f"  Loaded {len(records):,} records")

    if args.winrates:
        print("\nCard win rates (top 20):")
        rates = compute_card_win_rates(records)
        for card_id, win_rate in sorted(rates.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  Card {card_id:3d}: {win_rate:.1%}")
            
    print("\nBuilding dataset ...")
    states, actions, outcomes = build_dataset(records)

    print(f"  states  : {states.shape}  dtype={states.dtype}")
    print(f"  actions : {actions.shape}  dtype={actions.dtype}")
    print(f"  outcomes: {outcomes.shape}  dtype={outcomes.dtype}")
    print(f"  Win rate in dataset: {outcomes.mean():.1%}")

    np.savez_compressed(args.output, states=states, actions=actions, outcomes=outcomes)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
