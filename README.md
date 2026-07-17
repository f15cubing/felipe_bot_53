# Neural Chess Engine & Lichess Bot

A chess engine with a neural-network evaluation function and alpha-beta search,
deployed as a live bot on Lichess: [lichess.org/@/felipe_bot_53](https://lichess.org/@/felipe_bot_53).

## How it works

### Evaluation

Positions are scored by a multi-layer perceptron (778 → 256 → 128 → 1) rather than
hand-tuned piece-square tables. The 778-element input encodes piece bitboards, side
to move, castling rights, and material-balance features (see `tensorizor.py`). The
network was trained with Huber loss on ~200k positions (1800–2400 Elo) sampled from
the Lichess database.

### Search

Negamax with alpha-beta pruning, iterative deepening, and a transposition table
(`search.py`). A quiescence search extends captures at the leaves to avoid the
horizon effect, and check extensions deepen forcing lines.

### Endgames and openings

- 3–5 piece Syzygy tablebases for exact endgame play (optional; set `SYZYGY_PATH`).
- A JSON opening book (`book.json`) for the first few moves.

## Running locally

```bash
pip install -r requirements.txt
python train.py         # trains model.pth from chess_data3.npz (see tensorizor.py)
python uci_wrapper.py   # exposes the engine over UCI on stdin/stdout
```

`model.pth` is committed, so you can play without retraining.

## Deploying the bot 24/7

The engine runs CPU-only and fits on a small always-free cloud VM.

1. Provision an Ubuntu 22.04/24.04 host and SSH in.
2. Install and set up:
   ```bash
   git clone https://github.com/f15cubing/felipe_bot_53.git
   cd felipe_bot_53
   ./setup.sh
   ```
3. Create a Lichess OAuth token with the `bot:play` scope at
   <https://lichess.org/account/oauth/token/create>, then set it in
   `lichess-bot/config.yml` along with the engine paths:
   ```yaml
   token: "lip_yourRealTokenHere"
   engine:
     dir: "/home/ubuntu/felipe_bot_53"
     interpreter: "/home/ubuntu/felipe_bot_53/venv/bin/python"
   ```
4. Test once: `cd lichess-bot && ../venv/bin/python lichess-bot.py -v`
5. Run it as a service (restarts on crash, survives reboots):
   ```bash
   sudo cp felipe-bot.service /etc/systemd/system/felipe-bot.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now felipe-bot
   journalctl -u felipe-bot -f
   ```

## Credits

Lichess API integration uses the [lichess-bot](https://github.com/lichess-org/lichess-bot)
framework, vendored under `lichess-bot/`.
