import json
import random
import os
import chess

def load_opening_book():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    book_path = os.path.join(BASE_DIR, "book.json")

    if os.path.exists(book_path):
        with open(book_path, "r") as f:
            return json.load(f)
    return {}

def get_book_move(board, OPENING_BOOK):
    fen = board.fen()
    # Check if the current position is in our custom book
    if fen in OPENING_BOOK:
        move_list = OPENING_BOOK[fen]
        move_uci = random.choice(move_list)
        return chess.Move.from_uci(move_uci)
    return None
