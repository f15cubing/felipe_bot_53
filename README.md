# Neural Chess Engine & Lichess Bot
[![Lichess Bot](https://img.shields.io/badge/Lichess-Play%20Against%20Me-orange)](https://lichess.org/@/felipe_bot_53)

A custom-built chess engine featuring a neural network evaluation function and minimax search, deployed as a live bot on Lichess.

## 🤖 Play the Bot
The engine is currently hosted on a Google Cloud VM and accepts challenges on Lichess:
**[Play against the bot here](https://lichess.org/@/felipe_bot_53)**

---

## 🧠 Technical Overview

### 1. Neural Network Evaluation
Instead of traditional piece-square tables, this engine uses a **Multi-Layer Perceptron (MLP)** to evaluate positions.
- **Architecture:** 778 → 256 → 128 → 1.
- **Input:** 778-element feature tensors (representing bitboards for all piece types, side to move, castling rights, and en passant).
- **Training:** Trained on **200,000+ random positions** (1800-2400 Elo range) from the Lichess database.
- **Method:** Network trained using **Huber Loss** and **Gradient Descent** via **PyTorch**.

### 2. Search Algorithm
- **Negamax:** Implemented with **Alpha-Beta Pruning** to drastically reduce the search tree size.
- **Quiescence Search:** Prevents the "horizon effect" by continuing the search through tactical captures using a simplified material evaluator.
- **Transposition Table:** Utilizes a hash map to store previously evaluated positions, significantly increasing search depth per second.

### 3. Endgame & Openings
- **Syzygy Tablebases:** Integrated 3-5 piece tablebases for mathematically perfect play in the endgame.
- **Opening Book:** Uses a Polyglot-style opening book to ensure varied and rapid play in the early game.

---

## 🛠️ Infrastructure & Deployment
- **Cloud:** Hosted on a **Google Cloud Platform (GCP)** Compute Engine instance.
- **OS:** **Ubuntu Linux**, managed entirely via **SSH**.
- **Environment:** Production environment configured using **Linux CLI**, with code iterations handled via **Nano** and **Git**.
- **API:** Communicates with the Lichess API using the `python-chess` library and a custom UCI wrapper.

---

## 🚀 Local Setup
To run the engine locally:
1. Clone the repo: `git clone https://github.com/f15cubing/lichess-bot-felipe-53.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the trainer: `python train.py` (requires dataset)
4. Play via UCI: `python uci_wrapper.py`

## ⚖️ Credits & Licensing
- **API Wrapper:** This project utilizes the [lichess-org/lichess-bot](https://github.com/lichess-org/lichess-bot) framework to interface with the Lichess API.
