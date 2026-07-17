#!/usr/bin/env bash
# One-shot setup for the Neural Chess Engine & Lichess Bot.
#
# Tested on a fresh Ubuntu 22.04/24.04 host (e.g. an Oracle Cloud
# "Always Free" VM.Standard.A1.Flex / Ampere ARM instance).
#
# Usage:
#   git clone https://github.com/f15cubing/felipe_bot_53.git
#   cd felipe_bot_53
#   ./setup.sh
#
# After it finishes:
#   1. Put your Lichess OAuth token (scope: bot:play) in lichess-bot/config.yml
#      and point engine.dir / engine.interpreter at this directory (see below).
#   2. Test:      cd lichess-bot && ../venv/bin/python lichess-bot.py -v
#   3. Run 24/7:  see felipe-bot.service in this repo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ">> Installing system packages (python venv + git)..."
sudo apt-get update -y
sudo apt-get install -y python3-venv git

echo ">> Creating virtualenv..."
python3 -m venv venv

echo ">> Installing Python dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo ">> Done. Next steps:"
echo "   Edit lichess-bot/config.yml and set:"
echo "     token:       \"<your lichess bot:play token>\""
echo "     engine.dir:  \"$REPO_DIR\""
echo "     engine.interpreter: \"$REPO_DIR/venv/bin/python\""
echo ""
echo "   Test run:  cd lichess-bot && ../venv/bin/python lichess-bot.py -v"
echo "   Install as a service (edit paths/User first): see felipe-bot.service"
