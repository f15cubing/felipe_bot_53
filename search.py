import torch
import chess.syzygy
import time
import chess
import chess.polyglot
from train import NeuroChessNet
from tensorizor import board_to_tensor
from book_functions import get_book_move, load_opening_book
import os

EXACT = 0
LOWERBOUND = 1 # Beta cutoff
UPPERBOUND = 2 # Alpha (all-node)
OPENING_BOOK = load_opening_book()


class TranspositionTable:
    def __init__(self):
        self.table = {}

    def store(self, board_hash, depth, value, type, move):
        # We only overwrite if the new search was deeper
        if board_hash in self.table:
            if self.table[board_hash]['depth'] > depth:
                return
        self.table[board_hash] = {
            'depth': depth,
            'value': value,
            'type': type,
            'move': move
        }

    def probe(self, board_hash):
        return self.table.get(board_hash)

# Initialize global TT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

SYZYGY_PATH = "/home/f15cubing/syzygy"
tablebases = None

if os.path.exists(SYZYGY_PATH):
    try:
        tablebases = chess.syzygy.open_tablebase(SYZYGY_PATH)
        print(f"info string Loaded Syzygy tablebases from {SYZYGY_PATH}")
    except Exception as e:
        print(f"info string Failed to load tablebases: {e}")


tt = TranspositionTable()

torch.set_num_threads(1)

# Load the brain
model = NeuroChessNet()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
else:
    print(f"WARNING: Could not find model at {MODEL_PATH}")
model.eval()
for param in model.parameters():
    param.requires_grad = False # Saves memory by not tracking gradients

START_TIME = 0
TIME_LIMIT = 0
SEARCH_ABORTED = False

def get_nn_prediction(board):
    # Convert board to tensor and get model prediction
    tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # The model output is the score for the side-to-move
        return model(tensor).item()

def negamax(board, depth, alpha, beta):
    global SEARCH_ABORTED

    if tablebases and len(board.piece_map()) <= 5:
        try:
            # get_wdl returns 2 (win), 0 (draw), -2 (loss)
            res = tablebases.get_wdl(board)
            if res is not None:
                # Map -2 and 2 to high scores so engine prefers TB wins over NN scores
                # We use 90.0 so it's slightly less than actual checkmate (100.0)
                return res * 45.0 
        except Exception:
            pass # Fall back to search if probe fails

    if board.is_game_over():
        return evaluate_board(board)

    board_hash = chess.polyglot.zobrist_hash(board)
    tt_entry = tt.probe(board_hash)

    # Check if we have already seen this position
    if tt_entry and tt_entry['depth'] >= depth:
        tt_move = tt_entry.get('move')
        if tt_move is None or tt_move in board.legal_moves:
            if tt_entry['type'] == EXACT:
                return tt_entry['value']
            elif tt_entry['type'] == LOWERBOUND:
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['type'] == UPPERBOUND:
                beta = min(beta, tt_entry['value'])

            if alpha >= beta:
                return tt_entry['value']

    if (negamax.nodes & 127) == 0:
        if time.time() - START_TIME > TIME_LIMIT:
            SEARCH_ABORTED = True
            return alpha

    negamax.nodes += 1

    if board.is_game_over():
        return evaluate_board(board)

    if depth <= 0:
        return quiescence_search(board, alpha, beta)

    max_eval = -float('inf')
    original_alpha = alpha
    best_move = None

    hash_move = tt_entry['move'] if (tt_entry and tt_entry['move'] in board.legal_moves) else None

    # Get legal moves and sort them (simple ordering: captures first)
    moves = sorted(board.legal_moves, key=lambda m: (m==hash_move, get_move_priority(board, m)), reverse=True)

    for move in moves:
        board.push(move)
        extension = 1 if board.is_check() else 0
        eval = -negamax(board, depth - 1+extension, -beta, -alpha)
        board.pop()

        if SEARCH_ABORTED: return alpha # Stop searching immediately

        if eval > max_eval:
            max_eval = eval
            best_move = move

        alpha = max(alpha, eval)
        # Alpha-beta pruning
        if alpha >= beta:
            break

    if max_eval <= original_alpha:
        node_type = UPPERBOUND
    elif max_eval >= beta:
        node_type = LOWERBOUND
    else:
        node_type = EXACT

    tt.store(board_hash, depth, max_eval, node_type, best_move)

    return max_eval

negamax.nodes = 0

def select_best_move(board, time_limit_ms):
    tt.table.clear()
    book_move = get_book_move(board, OPENING_BOOK)
    if book_move and book_move in board.legal_moves:
        print(f"info string Playing move from custom opening book")
        return book_move

    # Check if we can use the tablebase
    if tablebases and len(board.piece_map()) <= 5:
        try:
            # DTZ (Distance to Zeroing) finds the fastest way to win/draw
            best_move = None
            best_dtz = float('inf')
            
            for move in board.legal_moves:
                board.push(move)
                # We want the move that results in the worst DTZ for the opponent
                dtz = tablebases.get_dtz(board) 
                board.pop()
                
                if dtz is not None:
                    if best_move is None or dtz < best_dtz:
                        best_dtz = dtz
                        best_move = move
            
            if best_move:
                print(f"info string Tablebase move found (DTZ: {best_dtz})")
                return best_move
        except Exception:
            pass

    global START_TIME, TIME_LIMIT, SEARCH_ABORTED
    negamax.nodes = 0
    START_TIME = time.time()
    TIME_LIMIT = time_limit_ms / 1000.0
    SEARCH_ABORTED = False
    if not board.legal_moves: return None
    best_move = list(board.legal_moves)[0] # Fallback move
    last_completed_best_move = best_move

    # Iterative Deepening
    for depth in range(1, 20): # Try up to depth 20
        negamax.nodes = 0
        move = search_at_depth(board, depth)

        if not SEARCH_ABORTED and move is not None:
            last_completed_best_move = move
        else:
            break

        # Since each level takes longer, if we've used most of the time already, we know we won't finish the next level
        if (time.time() - START_TIME) > (TIME_LIMIT * 0.6):
            break

    return last_completed_best_move

def search_at_depth(board, depth):
    best_move = None
    max_eval = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    board_hash = chess.polyglot.zobrist_hash(board)
    tt_entry = tt.probe(board_hash)
    hash_move = tt_entry['move'] if (tt_entry and tt_entry['move'] in board.legal_moves) else None

    # Sort moves, putting the hash_move first only if it's legal
    moves = sorted(board.legal_moves, 
                   key=lambda m: (m == hash_move, get_move_priority(board, m)), 
                   reverse=True)
    best_move = moves[0]

    for move in moves:
        board.push(move)
        eval = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if SEARCH_ABORTED:
            return best_move

        if eval > max_eval:
            max_eval = eval
            best_move = move
            alpha = max(alpha, eval)

    return best_move

def get_move_priority(board, move):
    """Assigns a simple numerical score to a move for sorting."""
    if board.is_capture(move):
        # MVV-LVA: (Value of Victim) - (Value of Attacker)
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        # Simple piece values for sorting
        values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
        v_val = values.get(victim.piece_type, 0) if victim else 0
        a_val = values.get(attacker.piece_type, 0) if attacker else 0

        return 100 + (v_val - a_val) # Captures get 100+ priority

    if move.promotion:
        return 90 # Promotions are high priority

    return 0 # Quiet moves

def quiescence_search(board, alpha, beta):
    stand_pat = evaluate_board(board, use_nn=False)
    if stand_pat >= beta: return beta
    if alpha < stand_pat: alpha = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    captures.sort(key=lambda m: get_move_priority(board, m), reverse=True)

    # Only look at "loud" moves, but limit them
    # Filter for captures that actually matter
    for move in captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha)
        board.pop()

        if score >= beta: return beta
        if score > alpha: alpha = score
    return alpha

def get_material_score(board):
    values = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0} # Scale to NN range
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = values[p.piece_type]
            # If it's White's turn, White pieces are positive.
            # If it's Black's turn, Black pieces are positive.
            score += val if p.color == board.turn else -val
    return score

def evaluate_board(board, use_nn = True):
    outcome = board.outcome()

    if outcome:
        if outcome.winner == board.turn: return 100.0
        if outcome.winner is not None: return -100.0
        # Draw Contempt: Prefer to keep playing if we are winning
        mat_score = get_material_score(board)
        return -1.0 if mat_score > 2.0 else 0.0

    mat_score = get_material_score(board)
    score = get_nn_prediction(board) if use_nn else 0.1 * mat_score

    # Clean up script when few pieces remain
    if abs(mat_score) > 2.0 and len(board.piece_map()) < 10:
        # Determine who is winning
        is_white_winning = mat_score > 0
        winning_color = chess.WHITE if is_white_winning else chess.BLACK
        
        loser_k = board.king(not winning_color)
        winner_k = board.king(winning_color)

        if loser_k is None or winner_k is None:
            return score

        mop_up_bonus = 0
        
        # 1. Push losing king to the edge/corners
        # (Distance from center: center squares are rank/file 3 and 4)
        file_dist = max(3 - chess.square_file(loser_k), chess.square_file(loser_k) - 4)
        rank_dist = max(3 - chess.square_rank(loser_k), chess.square_rank(loser_k) - 4)
        mop_up_bonus += (file_dist + rank_dist) * 0.1

        # 2. Reduce distance between the two kings
        dist_between_kings = chess.square_distance(winner_k, loser_k)
        mop_up_bonus += (14 - dist_between_kings) * 0.1
        
        if board.turn == winning_color:
            score += mop_up_bonus*0.1
        else:
            score -= mop_up_bonus*0.1

    return score

# --- TEST IT ---
if __name__ == "__main__":
    board = chess.Board()
    print("AI is thinking...")
    move = select_best_move(board, 10000)
    print(f"AI suggests move: {move}")
