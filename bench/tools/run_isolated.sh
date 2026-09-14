#!/bin/sh
# Runs every benchmark entry matching a filter in a process of its own.
#
# usage: run_isolated.sh [-c CPULIST] [-s SEED] BENCH FILTER REPEATS [EXTRA_ARGS...]
#
#   BENCH       the frsr_roaring_bench binary
#   FILTER      substring selecting entries, as for --filter
#   REPEATS     how many processes to start per entry, a positive integer
#   -c CPULIST  pin each process with taskset -c CPULIST (skipped with a warning when taskset is absent)
#   -s SEED     seed of the per-repeat entry order (default 1)
#   EXTRA_ARGS  passed to every process, e.g. --min-time-ms 200 --stat median
#
# The entries are taken from --list; each process is started with --exact --filter <entry>, so it sets up, warms and
# measures one arm of one band in a fresh address space and heap, with no earlier band's state inherited. Repeats are
# the outer loop and entries the inner one, so drift over the run (thermal, background load) lands on every entry
# evenly instead of on whichever entries come last. Each repeat walks the entries in its own pseudo-random order
# (seeded by SEED and the repeat number), so no arm of a band is always the one launched first.
#
# Prints each result line with a trailing "repeat=<n>" field. Exits 2 on a usage error and 1 when the listing fails or
# matches nothing. A process that exits non-zero or prints no result line for its entry (an "#aa" line from --aa does
# not count) is reported on stderr, the remaining runs go ahead, and the script exits non-zero at the end.

set -eu

me=run_isolated.sh
nl='
'
cr=$(printf '\r')

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2
}

cpus=
seed=1
while [ $# -gt 0 ]; do
    case $1 in
        -c) [ $# -ge 2 ] || { echo "$me: -c needs a CPU list" >&2; exit 2; }
            cpus=$2; shift 2 ;;
        -s) [ $# -ge 2 ] || { echo "$me: -s needs a seed" >&2; exit 2; }
            seed=$2; shift 2 ;;
        *)  break ;;
    esac
done
case $seed in
    ''|*[!0-9]*) echo "$me: SEED must be a non-negative integer, not '$seed'" >&2; exit 2 ;;
esac
if [ $# -lt 3 ]; then
    usage
    exit 2
fi
bench=$1
filter=$2
repeats=$3
shift 3
case $repeats in
    ''|*[!0-9]*|0*) echo "$me: REPEATS must be a positive integer, not '$repeats'" >&2; exit 2 ;;
esac

pin=
if [ -n "$cpus" ]; then
    if command -v taskset >/dev/null 2>&1; then
        pin="taskset -c $cpus"
    else
        echo "$me: taskset not found, running unpinned" >&2
    fi
fi

# --list prints a header and then one entry per line, indented by two spaces.
code=0
listing=$("$bench" --list </dev/null) || code=$?
if [ "$code" -ne 0 ]; then
    echo "$me: '$bench --list' failed with status $code" >&2
    exit 1
fi
entries=
old_ifs=$IFS
set -f
IFS=$nl
for line in $listing; do
    line=${line%"$cr"}
    case $line in
        "  "*) name=${line#  }
               case $name in *"$filter"*) entries=$entries$name$nl ;; esac ;;
    esac
done
IFS=$old_ifs
if [ -z "$entries" ]; then
    echo "$me: no entries match '$filter'" >&2
    exit 1
fi

status=0

# run_entry ENTRY REPEAT [EXTRA_ARGS...]
run_entry() {
    entry=$1
    rep=$2
    shift 2
    code=0
    out=$($pin "$bench" --exact --filter "$entry" "$@" </dev/null) || code=$?
    found=
    IFS=$nl
    for line in $out; do
        line=${line%"$cr"}
        case $line in
            "$entry	"*)     printf '%s\trepeat=%s\n' "$line" "$rep"; found=1 ;;
            "$entry#aa	"*)  printf '%s\trepeat=%s\n' "$line" "$rep" ;;
        esac
    done
    IFS=$old_ifs
    if [ "$code" -ne 0 ]; then
        echo "$me: '$entry' (repeat $rep) exited with status $code" >&2
        status=$code
    fi
    if [ -z "$found" ]; then
        echo "$me: '$entry' (repeat $rep) printed no result line" >&2
        [ "$status" -ne 0 ] || status=1
    fi
}

repeat=1
while [ "$repeat" -le "$repeats" ]; do
    # awk's srand/rand give the same sequence for the same seed on one system, which is all a run needs. Entry names
    # contain no spaces, so a space separates the sort key from the name.
    order=$(printf '%s' "$entries" | awk -v s=$((seed * 1000003 + repeat)) 'BEGIN { srand(s) } { print rand(), $0 }' | sort -k1,1 | cut -d' ' -f2-)
    IFS=$nl
    for entry in $order; do
        IFS=$old_ifs
        run_entry "$entry" "$repeat" "$@"
    done
    IFS=$old_ifs
    repeat=$((repeat + 1))
done
exit $status
