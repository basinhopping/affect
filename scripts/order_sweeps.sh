#!/usr/bin/env bash
# Run sweeps of equal priority one after another (added 2026-10-29). Run on the node once the sweeps are enqueued,
# for instance after smoke_enqueue.sh in the same tmux session:
#     bash ~/affect/scripts/order_sweeps.sh <sweep id> [<sweep id> ...]
# A pending job's file name sorts by priority, then by the job's number within its sweep, then by sweep ID
# (affect.sweep.queue.pending_name), so sweeps of equal priority take turns: job 0 of each, then job 1 of each, and so
# on. When a night ends before such a group is done, every sweep in it lacks its last jobs, and no record can be made
# of any of them. This script renames the pending files of the sweeps given so that, within each priority, all of the
# first sweep's jobs come first, then all of the second's, in the order given: the six-digit number in each name
# becomes the sweep's place in the list (1 to 99) times 10,000 plus the job's own number. The priority part of the
# name, the pending files of every other sweep (which keep their places, before these) and the content of every job
# stay as they are; a worker reads its job from the file, not from the name. Safe while workers run: a rename is
# atomic, and a file a worker takes first is simply not renamed (its mv finds no file). Running it again with the
# same list changes nothing. A sweep not enqueued (no artifacts/sweeps/<id>/jobs.csv) or with 10,000 jobs or more is
# left as it is. Prints, per sweep, the files renamed, those already in place and those taken by a worker meanwhile,
# then ORDERED with the total and the first pending file.
set -uo pipefail
ROOT="${AFFECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 1
[ "$#" -gt 0 ] || { echo "usage: order_sweeps.sh <sweep id> [<sweep id> ...]"; exit 1; }
[ "$#" -le 99 ] || { echo "at most 99 sweeps, not $#"; exit 1; }
P=queue/pending
total=0
place=0
for s in "$@"; do
    place=$((place + 1))
    if [ ! -f "artifacts/sweeps/$s/jobs.csv" ]; then echo "$s: not enqueued (no artifacts/sweeps/$s/jobs.csv), left as it is"; continue; fi
    jobs=$(($(wc -l < "artifacts/sweeps/$s/jobs.csv") - 1))
    if [ "$jobs" -ge 10000 ]; then echo "$s: $jobs jobs, 10,000 or more, left as it is"; continue; fi
    renamed=0; in_place=0; taken=0; odd=0
    for f in "$P"/*__"$s"__*.json; do
        [ -e "$f" ] || continue
        b=${f##*/}
        if ! [[ "$b" =~ ^([0-9]{3})_([0-9]{6})__"$s"__(.+)$ ]]; then odd=$((odd + 1)); continue; fi
        new=$(printf '%s_%02d%04d__%s__%s' "${BASH_REMATCH[1]}" "$place" "$((10#${BASH_REMATCH[2]} % 10000))" "$s" \
            "${BASH_REMATCH[3]}")
        if [ "$new" = "$b" ]; then in_place=$((in_place + 1)); continue; fi
        if [ -e "$P/$new" ]; then odd=$((odd + 1)); continue; fi
        if mv -n "$f" "$P/$new" 2>/dev/null && [ -e "$P/$new" ]; then renamed=$((renamed + 1)); else taken=$((taken + 1)); fi
    done
    total=$((total + renamed))
    echo "$s: place $place, $renamed renamed, $in_place already in place, $taken taken by a worker meanwhile$( \
        [ "$odd" = 0 ] || echo ", $odd left as they are (name not of the queue's form, or its new name taken)")"
done
echo "ORDERED $(date -u +%FT%TZ): $total pending files renamed; first pending $(LC_ALL=C ls "$P" | head -1)"
