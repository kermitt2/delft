#!/bin/bash
# Streams every candidate through the filter. Low priority, three streams at a time,
# stops starting new ones when the disk gets short. Re-runnable: finished parts are skipped.
cd "$(dirname "$0")"; mkdir -p parts logs
LANGS="en fr de es it pt nl ru zh ja ko ar"
guard() { [ "$(df --output=avail -BG . | tail -1 | tr -dc 0-9)" -ge 15 ]; }
one() { # name url gz
  [ -f parts/$1.done ] && return 0
  guard || { echo "disk guard: skip $1"; return 1; }
  if [ "$3" = gz ]; then dec="gunzip -c"; else dec="cat"; fi
  curl -sSfL --retry 3 --max-time 7200 "$2" | $dec | nice -n 19 python3 filter_stream.py data/vocab.txt "parts/$1.LANG.tmp" \
    && [ "${PIPESTATUS[0]}" -eq 0 ] && mv parts/$1.x.tmp parts/$1.vec && touch parts/$1.done || echo "FAILED $1"
}
export -f one guard
{
  for l in $LANGS; do echo "aligned_$l https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.$l.align.vec plain"; done
  for l in $LANGS; do echo "cc_$l https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$l.300.vec.gz gz"; done
} | xargs -P 3 -L 1 bash -c 'one "$0" "$1" "$2"'
if [ ! -f parts/numberbatch.done ] && guard; then
  curl -sSfL --retry 3 https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz | gunzip -c \
   | nice -n 19 python3 filter_stream.py data/vocab.txt parts/numberbatch_LANG.vec --numberbatch && touch parts/numberbatch.done
fi
echo ALL_STREAMS_DONE
