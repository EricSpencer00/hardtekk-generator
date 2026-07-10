#!/bin/zsh
# Watches input/ — drop any song file there and a hardtekk wav appears in output/.
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/input" "$DIR/output" "$DIR/.done"
echo "Watching $DIR/input — drop songs there. Ctrl-C to stop."
while true; do
  for f in "$DIR"/input/*.(mp3|wav|m4a|flac|ogg|aac)(N); do
    base="${f:t}"
    [[ -e "$DIR/.done/$base" ]] && continue
    # wait until file size is stable (finished copying)
    s1=$(stat -f%z "$f"); sleep 1; s2=$(stat -f%z "$f")
    [[ "$s1" != "$s2" ]] && continue
    echo ">> $base"
    "$DIR/.venv/bin/python" "$DIR/hardtekk.py" "$f" && touch "$DIR/.done/$base"
  done
  sleep 2
done
