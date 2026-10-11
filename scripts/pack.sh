#!/usr/bin/env bash
# Pack finished results for the PC. Run on the node:
#     bash ~/affect/scripts/pack.sh checks            (everything in artifacts/checks)
#     bash ~/affect/scripts/pack.sh runs/<sweep_id>   (one sweep)
#     cd ~/affect/artifacts && bash ~/affect/scripts/pack.sh runs/cnn_a runs/cnn_b sweeps/cnn_a sweeps/cnn_b
#                                                     (several folders in one archive, one pull)
#     bash ~/affect/scripts/pack.sh --after <earlier archive name> runs/<sweep_id>
#                                                     (only files that earlier archive did not hold)
#     bash ~/affect/scripts/pack.sh --after <first archive> --after <second archive> runs/<sweep_id>
#                                                     (only files none of those archives held)
# Writes ~/affect_outgoing/<name>.tar.gz and <name>.tar.gz.sha256, where <name> is printed at
# the end; scripts/pull.ps1 <name> fetches and checks them. The archive holds a list of every
# file's SHA-256, which pull.ps1 checks after unpacking. Refuses while any run folder under the
# given folders is unfinished (*.tmp). With several folders the name is the first one's plus
# "and_<n>_more"; the list of files inside says exactly what is there.
# --after is for a sweep pulled before it finished: the earlier archive (still in
# ~/affect_outgoing) lists what the PC already has, so only the rest is packed and pull.ps1, which
# refuses to overwrite, can unpack it. A file the earlier archive holds with other content stops it.
# A sweep pulled in more than two parts needs every earlier part's archive: --after may be given
# several times (added 2026-09-30, for the LIF screens, which run over several nights), and the
# PC then has the files of all of them.
set -euo pipefail
source "$(dirname "$0")/common.sh"
USAGE="usage: pack.sh [--after <earlier archive name>]... <path under artifacts/> [more paths under artifacts/]"
AFTER=()
NA=0      # how many --after: a plain count, since old bash versions refuse an empty "${AFTER[@]}" under set -u
while [ "${1:-}" = "--after" ]; do
    AFTER+=("${2:?$USAGE}")
    NA=$((NA + 1))
    shift 2
done
[ "$#" -ge 1 ] || { echo "$USAGE"; exit 1; }
cd "$AFFECT_ROOT/artifacts"
for SUB in "$@"; do
    test -d "$SUB" || { echo "no artifacts/$SUB"; exit 1; }
    # -quit: find stops at the first match, so nothing is left writing into a closed pipe
    if [ -n "$(find "$SUB" -type d -name "*.tmp" -print -quit)" ]; then
        echo "REFUSING: unfinished run folders (*.tmp) under artifacts/$SUB"; exit 1
    fi
done
OUT="$HOME/affect_outgoing"
for A in ${AFTER[@]+"${AFTER[@]}"}; do
    if [ ! -f "$OUT/$A.tar.gz" ]; then echo "no earlier archive $OUT/$A.tar.gz"; exit 1; fi
done
LABEL="$(echo "$1" | tr '/' '_')"
if [ "$#" -gt 1 ]; then LABEL="${LABEL}_and_$(($# - 1))_more"; fi
if [ "$NA" -gt 0 ]; then LABEL="${LABEL}_new"; fi
NAME="affect_results_${LABEL}_$(stamp)"
STAGE="$OUT/$NAME.stage"
# Whatever stops the script (an error, a dropped connection), the stage folder goes, and so does an
# archive that never got its .sha256 (added 2026-09-30).
DONE=0
trap 'rm -rf "$STAGE"; [ "$DONE" = 1 ] || rm -f "$OUT/$NAME.tar.gz" "$OUT/$NAME.tar.gz.sha256"' EXIT
mkdir -p "$STAGE/_pulled"
LIST="$STAGE/_pulled/$NAME.files.sha256"
if [ "$NA" -eq 0 ]; then
    find "$@" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$LIST"
    tar -czf "$OUT/$NAME.tar.gz" -C "$AFFECT_ROOT/artifacts" "$@" -C "$STAGE" _pulled
else
    : > "$STAGE/earlier.sha256"
    for A in "${AFTER[@]}"; do
        tar -xzOf "$OUT/$A.tar.gz" "_pulled/$A.files.sha256" >> "$STAGE/earlier.sha256" \
            || { echo "no list of files in $A.tar.gz"; rm -rf "$STAGE"; exit 1; }
    done
    find "$@" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$STAGE/now.sha256"
    # A sha256sum line is 64 hex digits, two characters, then the path (from column 67).
    awk -v changed="$STAGE/changed.txt" '
        FILENAME == ARGV[1] { earlier[substr($0, 67)] = substr($0, 1, 64); next }
        { path = substr($0, 67)
          if (!(path in earlier)) print
          else if (earlier[path] != substr($0, 1, 64)) print path > changed }' \
        "$STAGE/earlier.sha256" "$STAGE/now.sha256" > "$LIST"
    if [ -s "$STAGE/changed.txt" ]; then
        echo "REFUSING: these files changed since ${AFTER[*]} (the PC has the earlier content):"
        head -20 "$STAGE/changed.txt"
        rm -rf "$STAGE"; exit 1
    fi
    if [ ! -s "$LIST" ]; then
        echo "nothing new under $* since ${AFTER[*]}"
        rm -rf "$STAGE"; exit 1
    fi
    cut -c67- "$LIST" > "$STAGE/names.txt"
    tar -czf "$OUT/$NAME.tar.gz" -C "$AFFECT_ROOT/artifacts" -T "$STAGE/names.txt" -C "$STAGE" _pulled
fi
rm -rf "$STAGE"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
DONE=1
echo "packed $(wc -l < <(tar -tzf "$OUT/$NAME.tar.gz")) entries from $# folders"
echo "On the PC: powershell -ExecutionPolicy Bypass -File \"C:\\Affective Computing\\affect\\scripts\\pull.ps1\" $NAME"
