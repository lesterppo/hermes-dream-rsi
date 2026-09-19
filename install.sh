#!/usr/bin/env bash
# Install the dream-rsi CLI (and optionally the Hermes tool plugin).
#
#   ./install.sh              # CLI + symlink
#   ./install.sh --plugin     # also install the Hermes native tool
#   ./install.sh --check      # report what is installed, change nothing
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/bin}"
DEST="${DREAMRSI_HOME:-$HOME/.hermes/scripts/dream_rsi}"
PLUGIN_DIR="${HERMES_HOME:-$HOME/.hermes}/plugins/hermes_local_tools"
MODE="${1:-}"

report() {
  echo "dream-rsi: $1"
}

if [ "$MODE" = "--check" ]; then
  command -v python3 >/dev/null && report "python3 ok ($(python3 -V 2>&1))" || report "python3 MISSING"
  python3 -c "import numpy" 2>/dev/null && report "numpy ok" || report "numpy MISSING (pip install numpy)"
  [ -x "$PREFIX/dream-rsi" ] && report "CLI installed at $PREFIX/dream-rsi" || report "CLI not installed"
  [ -f "$PLUGIN_DIR/dream_rsi_tool.py" ] && report "Hermes tool installed" || report "Hermes tool not installed"
  exit 0
fi

python3 -c "import numpy" 2>/dev/null || { echo "ERROR: numpy required (pip install numpy)"; exit 1; }

mkdir -p "$DEST" "$PREFIX"
if [ "$SRC" != "$DEST" ]; then
  for item in dream_rsi see bin tests README.md AGENTS.md; do
    [ -e "$SRC/$item" ] && cp -r "$SRC/$item" "$DEST/"
  done
fi
chmod +x "$DEST/bin/dream-rsi"
ln -sf "$DEST/bin/dream-rsi" "$PREFIX/dream-rsi"
report "CLI -> $PREFIX/dream-rsi (package at $DEST)"

if [ "$MODE" = "--plugin" ]; then
  if [ -d "$PLUGIN_DIR" ]; then
    cp "$SRC/hermes_tool/dream_rsi_tool.py" "$PLUGIN_DIR/dream_rsi_tool.py"
    report "Hermes tool -> $PLUGIN_DIR/dream_rsi_tool.py (restart Hermes to load)"
  else
    report "Hermes plugin dir not found at $PLUGIN_DIR - skipping tool install"
  fi
fi

report "smoke test:"
"$PREFIX/dream-rsi" tasks >/dev/null && report "  CLI responds" || report "  CLI FAILED"
echo
echo "Next:  dream-rsi init --run ~/runs/demo --task circle_packing"
echo "       dream-rsi loop --run ~/runs/demo --task circle_packing --agent deepseek --rounds 2"
