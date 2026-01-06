import chess
import numpy as np

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Black
}

def board_to_tensor(board):
    # Create an array of 778 zeros
    tensor = np.zeros(778, dtype=np.float32)
    # Determine if we need to flip perspective
    is_white = board.turn == chess.WHITE

    # The first 768 numbers are derived from (12 piece types * 64 squares).
    # A '1' indicates the presence of that specific piece at that square.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            p_type = piece.piece_type - 1 # 0 to 5

            if is_white:
                actual_plane = p_type if piece.color == chess.WHITE else p_type + 6
                sq_idx = square
            else:
                actual_plane = p_type if piece.color == chess.BLACK else p_type + 6
                sq_idx = square ^ 56 # Flip board vertically so Black "plays up"

            tensor[actual_plane * 64 + sq_idx] = 1

    # Side to move
    tensor[768] = 1 if is_white else -1

    white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in zip(range(1, 6), [1, 3, 3, 5, 9]))
    black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in zip(range(1, 6), [1, 3, 3, 5, 9]))
    rel_balance = (white_mat - black_mat) if is_white else (black_mat - white_mat)

    tensor[769] = np.clip(rel_balance / 39, -1, 1) # Material Balance
    tensor[770] = get_relative_balance(board, chess.PAWN, 8)                  # Pawn Balance
    tensor[771] = (get_relative_balance(board, chess.KNIGHT, 2) + get_relative_balance(board, chess.BISHOP, 2)) / 2 # Minor Piece Balance
    tensor[772] = get_relative_balance(board, chess.ROOK, 2)                  # Rook Balance
    tensor[773] = get_relative_balance(board, chess.QUEEN, 1)                 # Queen Balance

    # Castling Rights (Indices 774 - 777)
    if is_white:
        tensor[774] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        tensor[775] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        tensor[776] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        tensor[777] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    else:
        # Swap so Active player's rights are always first
        tensor[774] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        tensor[775] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        tensor[776] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        tensor[777] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0

    return tensor

def save_dataset(positions_with_scores, filename="chess_data.npz"):
    X = [] # Tensors
    y = [] # Scores

    for fen, score in positions_with_scores:
        board = chess.Board(fen)
        X.append(board_to_tensor(board))
        y.append(score)

        mirrored = board.mirror()
        X.append(board_to_tensor(mirrored))
        y.append(-score)
    # Save as compressed numpy file
    np.savez_compressed(filename, x=np.array(X), y=np.array(y))
    print(f"Saved {len(X)} positions to {filename}")

def process_checkpoint(input_file="raw_checkpoint.npz", output_file="chess_data3.npz"):
    loader = np.load(input_file)
    fens = loader['fens']
    scores = loader['scores']

    X = []
    y = []

    print(f"Tensorizing {len(fens)} positions...")

    for i in range(len(fens)):
        board = chess.Board(fens[i])
        score = scores[i]

        if abs(score) > 5000:
            score = np.sign(score) * 1000
        if board.turn == chess.BLACK:
            score = -score
        X.append(board_to_tensor(board))
        y.append(score)

        mirrored = board.mirror()
        X.append(board_to_tensor(mirrored))
        y.append(score)

        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}...")


    np.savez_compressed(output_file, x=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32))
    print(f"Final training data saved to {output_file}")

def get_relative_balance(board, piece_type, max_val):
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        res = (w - b) if board.turn == chess.WHITE else (b - w)
        return res / max_val

if __name__ == "__main__":
    process_checkpoint()
