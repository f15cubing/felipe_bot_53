#!/usr/bin/env python3
import sys
import chess
from search import select_best_move

def uci_loop():
    board = chess.Board()
    depth = 3 # Default depth

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        tokens = line.strip().split()
        if not tokens:
            continue

        command = tokens[0]

        if command == "uci":
            print("id name felipe_bot_53_v1")
            print("id author Felipe Caicedo")
            print("uciok", flush=True)

        elif command.startswith("setoption"):
            # Lichess tries to set things like 'Move Overhead'.
            # We just ignore these so the bot doesn't crash.
            continue

        elif command == "isready":
            print("readyok", flush =True)

        elif command == "ucinewgame":
            board = chess.Board()

        elif command == "position":
            try:
                if "startpos" in tokens:
                    board = chess.Board()
                elif "fen" in tokens:
                    # Find where FEN starts and where 'moves' starts
                    fen_start = tokens.index("fen") + 1
                    # If 'moves' exists, FEN is everything between 'fen' and 'moves'
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves")
                        fen_str = " ".join(tokens[fen_start:moves_idx])
                    else:
                        fen_str = " ".join(tokens[fen_start:])
                    board = chess.Board(fen_str)

                # Now handle moves regardless of how the position started
                if "moves" in tokens:
                    move_start = tokens.index("moves") + 1
                    for move in tokens[move_start:]:
                        try:
                            board.push_uci(move)
                        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                            # Skip illegal moves to prevent the crash
                            continue 
            except Exception as e:
                # Log the error if you have logging, or just reset to startpos 
                # to keep the engine from dying (Exit Code 1)
                board = chess.Board()

        elif command == "go":
            try:
                # 1. Extract values safely with defaults
                wtime = int(next((tokens[i+1] for i, t in enumerate(tokens) if t == "wtime"), 300000))
                btime = int(next((tokens[i+1] for i, t in enumerate(tokens) if t == "btime"), 300000))
                winc = int(next((tokens[i+1] for i, t in enumerate(tokens) if t == "winc"), 0))
                binc = int(next((tokens[i+1] for i, t in enumerate(tokens) if t == "binc"), 0))

                # 2. Pick the correct side's time
                my_time = wtime if board.turn == chess.WHITE else btime
                my_inc = winc if board.turn == chess.WHITE else binc

                # 3. Calculate time (Ensure it's a whole number for the search)
                allocated_time = int((my_time / 40) + (my_inc / 2))
                # Never use less than 100ms, never use more than what we have left minus a safety buffer
                allocated_time = int(max(100, min(allocated_time, my_time - 500)))

                # 4. Call search (This is where the crash usually happens)
                best_move = select_best_move(board, allocated_time)
                
                # 5. Output response safely
                if best_move:
                    # If it's a chess.Move object, use .uci(), otherwise print as string
                    move_str = best_move.uci() if hasattr(best_move, "uci") else str(best_move)
                    print(f"bestmove {move_str}", flush=True)
                else:
                    # Fallback if search returns None
                    print(f"bestmove {list(board.legal_moves)[0].uci()}", flush=True)

            except Exception as e:
                # CRITICAL: This keeps the process alive if search.py has a bug
                # It will pick the first legal move so the game continues
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}", flush=True)
                else:
                    print("bestmove 0000", flush=True)

        elif command == "quit":
            break

if __name__ == "__main__":
    uci_loop()
