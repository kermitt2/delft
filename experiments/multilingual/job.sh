#!/bin/bash
# job.sh <corpus> <embedding|none> <fold>: train on the fold, evaluate each stratum. Skips what is done.
corpus=$1; emb=$2; fold=$3
cd "$(dirname "$0")/work"
tag="$corpus.$emb.fold$fold"; out=../runs/$corpus/$emb/fold$fold; res=../results/$tag; mkdir -p $out $res
[ -f $res/DONE ] && exit 0
[ "$(df --output=avail -BG . | tail -1 | tr -dc 0-9)" -ge 8 ] || { echo "SKIP(disk) $tag"; exit 0; }
export PYTHONPATH=/data/workspace/lfoppiano/delft/delft-multiling-emb OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
PY=/home/lfoppian/miniforge3/envs/delft-onnx/bin/python
args=""; [ "$emb" != none ] && args="--embedding $emb"
start=$(date +%s)
timeout 7h nice -n 10 $PY -m delft.applications.grobidTagger $corpus train --architecture BidLSTM_CRF_FEATURES \
  --input ../data/$corpus/fold$fold/train.txt $args --output $out --seed $((fold+1)) --num-workers 2 > $res/train.log 2>&1
rc=$?
[ $rc -eq 0 ] || { echo "FAILED(train rc=$rc) $tag"; exit 0; }
echo $(( $(date +%s)-start )) > $res/train_seconds
model=$out/$corpus-BidLSTM_CRF_FEATURES
for t in ../data/$corpus/fold$fold/test.*.txt; do
  s=$(basename $t .txt); s=${s#test.}
  nice -n 10 $PY -m delft.applications.grobidTagger $corpus eval --architecture BidLSTM_CRF_FEATURES --input $t --input-model $model 2>&1 | grep -v 'it/s' > $res/eval.$s.log \
    || echo "FAILED(eval) $tag $s"
done
touch $res/DONE; echo "ok $tag $(cat $res/train_seconds)s"
