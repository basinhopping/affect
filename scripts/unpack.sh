#!/usr/bin/env bash
# Install an archive sent by scripts/push.ps1. Run on the node:
#     bash ~/affect_incoming/unpack.sh affect_code_<utc>.tar.gz
#     bash ~/affect_incoming/unpack.sh affect_data_iemocap_<utc>.tar
#     bash ~/affect_incoming/unpack.sh affect_code_<utc>.tar.gz ~/affect_next
# Every archive's SHA-256 is checked first.
# Code: src/, notebooks/, scripts/, tests/ and the top-level files in ~/affect are replaced
# wholesale, so a file removed on the PC cannot survive here. data/, artifacts/, queue/ and
# logs/ are never touched. Each install is recorded in <root>/CODE_VERSIONS.log.
# Since 2026-10-26 the old code is moved aside (renamed into ~/affect_incoming/replaced/<utc>_<root>/), not
# deleted, and the new code then moved in: /home on the node is NFS, where a file a running process holds open
# cannot be deleted (it becomes an .nfs placeholder that keeps its folder from being removed), which stopped the
# install of 2026-10-24 half way; a rename is not stopped by it. Copies moved aside by earlier installs are deleted
# at the start of the next install where nothing holds them open, and left (with a note) where something does. If
# an old item cannot be moved aside, the items already moved go back and nothing changes.
# Data: unpacked into ~/affect/data/<dataset>/assembled/, and refused if that folder exists.
# The optional second argument installs code into another folder (added 2026-09-29), for testing
# new code in ~/affect_next while workers run from ~/affect (its data/ and artifacts/ are links to
# ~/affect's). Installing into ~/affect is refused while a gpu tmux session exists: a running worker
# keeps the code it started with, and a module it imports later would come from the new files.
set -euo pipefail
IN="$HOME/affect_incoming"
NAME="${1:?usage: unpack.sh <archive name in ~/affect_incoming> [code folder, default ~/affect]}"
ROOT="${2:-$HOME/affect}"
ROOT="${ROOT%/}"
if [ "$ROOT" = "$HOME/affect" ] && tmux ls 2>/dev/null | grep -q '^gpu'; then
    echo "REFUSING: gpu sessions are running. touch ~/affect/queue/STOP, wait until tmux ls lists no gpu session, then run this again."
    exit 1
fi
cd "$IN"
sha256sum -c "$NAME.sha256"
STAGE="$IN/unpacking_$NAME"
mkdir "$STAGE"
trap 'rm -rf "$STAGE"' EXIT      # the staging folder goes however the install ends (added 2026-10-26)

case "$NAME" in
affect_code_*.tar.gz)
    tar -xzf "$NAME" -C "$STAGE"
    test -f "$STAGE/src/affect/__init__.py" || { echo "archive has no src/affect"; exit 1; }
    mkdir -p "$ROOT"
    ITEMS="src notebooks scripts sweeps tests README.md HANDOFF.md VERIFICATION_LOG.md findings.csv requirements.in
           .gitignore .gitattributes"
    OLD="$IN/replaced"
    mkdir -p "$OLD"
    for previous in "$OLD"/*; do
        if [ -e "$previous" ]; then
            rm -rf "$previous" 2>/dev/null || echo "note: $previous is still held open somewhere; left for a later install"
        fi
    done
    ASIDE=$(mktemp -d "$OLD/$(date -u +%Y%m%dT%H%M%SZ)_$(basename "$ROOT")_XXXX")
    moved=""
    for item in $ITEMS; do
        if [ -e "$ROOT/$item" ] || [ -L "$ROOT/$item" ]; then
            if mv "$ROOT/$item" "$ASIDE/$item"; then
                moved="$moved $item"
            else
                for back in $moved; do mv "$ASIDE/$back" "$ROOT/$back"; done
                rm -rf "$STAGE"
                echo "REFUSING: $ROOT/$item could not be moved aside; the code in $ROOT is unchanged"
                exit 1
            fi
        fi
    done
    for item in $ITEMS; do
        if [ -e "$STAGE/$item" ]; then mv "$STAGE/$item" "$ROOT/$item"; fi
    done
    find "$ROOT/scripts" -name "*.sh" -exec sed -i 's/\r$//' {} +
    left=$(find "$STAGE" -mindepth 1 -maxdepth 1)
    if [ -n "$left" ]; then echo "WARNING, not installed: $left"; fi
    rm -rf "$STAGE"
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$(cut -d' ' -f1 "$NAME.sha256")" "$NAME" >> "$ROOT/CODE_VERSIONS.log"
    echo "code in $ROOT replaced from $NAME"
    ;;
affect_data_*.tar)
    DATASET=$(echo "$NAME" | sed -E 's/^affect_data_([a-z0-9_]+)_[0-9TZ]+\.tar$/\1/')
    DEST="$ROOT/data/$DATASET/assembled"
    if [ -e "$DEST" ]; then echo "REFUSING: $DEST exists. Move it aside first."; rm -rf "$STAGE"; exit 1; fi
    tar -xf "$NAME" -C "$STAGE"
    test -f "$STAGE/assembled/MANIFEST.csv" || { echo "archive has no assembled/MANIFEST.csv"; exit 1; }
    mkdir -p "$ROOT/data/$DATASET"
    mv "$STAGE/assembled" "$DEST"
    rm -rf "$STAGE"
    echo "data in $DEST. Check it: bash $ROOT/scripts/check.sh"
    ;;
*)
    echo "unknown archive name: $NAME"; rm -rf "$STAGE"; exit 1 ;;
esac
