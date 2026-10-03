#!/bin/bash
# Submit a test matrix of DeLFT to a SLURM cluster as one throttled job array: every
# application, with every architecture, trained, evaluated, trained with n folds and used.
#
# Usage:
#   test_distributed_array.sh {smoke|full} [GROUP ...]
#   test_distributed_array.sh report LOG_DIR
#   test_distributed_array.sh __task ...   (what sbatch runs in every array task)
#
# `smoke` limits every training to 3 epochs and the n-fold trainings to 2 folds: it tells
# whether everything runs. `full` trains with the epochs of every application and 5 folds:
# it also gives scores to look at. `report` summarises the logs of a run.
#
# A task is one model with one architecture, and runs a sequence of steps:
#   train_eval  train on a part of the data and evaluate on the rest
#   nfold       the same with several folds
#   train       train on all the data
#   eval        evaluate the model `train` saved
#   tag         label or classify a few sample texts with that model
# An application runs the steps it has. A task goes on after a failed step, apart from
# `eval` and `tag` which need the model of `train`, and fails when any of its steps did.
#
# The groups are the applications: grobid (one row per GROBID model), ner, insult, dataset,
# citation, dataseer, license, software, software-context and toxic. All of them by default;
# a group whose training data is not in the checkout is left out, with a note.
#
# The matrix is rows x architectures. The embeddings (for the architectures without a
# transformer) and the transformers (for the BERT ones) are not a dimension: they rotate
# over the tasks, so that every one of them meets every architecture as the rows go by.
# PRODUCT=true runs every architecture with every one of them instead.
#
# Every task works in a directory of its own, where the data of the checkout is linked and
# the models are saved: a run never touches the models of data/models, and two tasks never
# write the same model. That directory is removed when the task ends.
#
# Environment overrides:
#   MODELS                     GROBID models (default: those with training data)
#   ARCHITECTURES              sequence labelling architectures (default: all)
#   CLASSIFIER_ARCHITECTURES   text classification architectures (default: all)
#   EMBEDDINGS                 embeddings to rotate over, `none` for no pre-trained ones
#   TRANSFORMERS               transformers to rotate over
#   MOST_EMBED_SCI             local copy of scilons/most-embed-sci, an embedding model used as
#                              contextual embeddings (default: under /netscratch). Without
#                              that directory the model is taken from the hub, where it is
#                              private: export HF_ACCESS_TOKEN, the tasks inherit it.
#   STEPS                      steps to run (default: train_eval nfold train eval tag)
#   MAX_EPOCH                  epochs of a training (default: 3 for smoke, unset for full)
#   FOLD_COUNT                 folds of the nfold step (default: 2 for smoke, 5 for full)
#   PRODUCT                    when "true", every embedding and transformer with every architecture
#   KEEP_MODELS                when "true", keep the working directories and their models
#   WORK_ROOT                  where the working directories go (default: data/test-runs/<run>)
#   MAX_PARALLEL_JOBS, ARRAY_SPEC, DRY_RUN, LOG_DIR, PYTHON_BIN and the cluster settings
#                              (CONTAINER_IMAGE, CONTAINER_WORKDIR, CONTAINER_MOUNTS, PARTITIONS,
#                              CPUS_PER_TASK, MEMORY, TIME_LIMIT, SBATCH_EXTRA): as for
#                              train_distributed_array.sh, see doc/distributed_training.md

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE-/netscratch/lfoppiano/enroot/delft-pytorch.sqsh}
# the tasks run the code of the checkout this script is part of, unless told otherwise
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/netscratch:/netscratch,$HOME:$HOME"}
PARTITIONS=${PARTITIONS:-RTX3090,RTXA6000,RTXB6000,L40S}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEMORY=${MEMORY:-100G}
TIME_LIMIT=${TIME_LIMIT:-1-00:00}
SBATCH_EXTRA=${SBATCH_EXTRA:-}

MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
DRY_RUN=${DRY_RUN:-false}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
PRODUCT=${PRODUCT:-false}
KEEP_MODELS=${KEEP_MODELS:-false}

ALL_GROUPS=(grobid ner insult dataset citation dataseer license software software-context toxic)
ALL_STEPS=(train_eval nfold train eval tag)

GROBID_ARCHITECTURES=(BidLSTM BidLSTM_CRF BidLSTM_ChainCRF BidLSTM_CNN_CRF BidGRU_CRF BidLSTM_CNN
                      BidLSTM_CRF_CASING BidLSTM_CRF_FEATURES BidLSTM_ChainCRF_FEATURES
                      BERT BERT_FEATURES BERT_CRF BERT_ChainCRF BERT_CRF_FEATURES
                      BERT_ChainCRF_FEATURES)
DEFAULT_CLASSIFIER_ARCHITECTURES=(lstm bidLstm_simple cnn cnn2 cnn3 lstm_cnn gru gru_simple gru_lstm
                                  dpcnn bert)
MOST_EMBED_SCI=${MOST_EMBED_SCI:-/netscratch/lfoppiano/delft/embeddings/most-embed-sci}
[[ -d "$MOST_EMBED_SCI" ]] || MOST_EMBED_SCI=scilons/most-embed-sci
# static embeddings (compiled and not) and contextual ones, from the registry and outside it
DEFAULT_EMBEDDINGS=(none glove-840B potion-base-8M static-retrieval-mrl-en scibert-contextual
                    "contextual:$MOST_EMBED_SCI")
DEFAULT_TRANSFORMERS=(
    allenai/scibert_scivocab_cased
    allenai/scibert_scivocab_uncased
    answerdotai/ModernBERT-base
    microsoft/deberta-v3-base
    michiyasunaga/LinkBERT-base
)

usage() {
    cat >&2 <<USAGE
Usage: $0 {smoke|full} [GROUP ...]     groups: ${ALL_GROUPS[*]}
       $0 report LOG_DIR
USAGE
    exit 2
}

# The lists of a run in the global LIST: the space-separated one given in the environment
# when there is one, else the default.
list_from_env() {
    local from_env=$1
    shift
    if [[ -n "$from_env" ]]; then
        read -r -a LIST <<<"$from_env"
    else
        LIST=("$@")
    fi
}

contains() {
    local wanted=$1 value
    shift
    for value in "$@"; do
        [[ "$value" != "$wanted" ]] || return 0
    done
    return 1
}

# The latest training file of a GROBID model, as grobidTagger finds it: the last date of
# the names <model>-YYMMDD.train[.gz], the plain file rather than the compressed one.
latest_train_file() {
    local model=$1 file latest="" latest_date="" date
    for file in "data/sequenceLabelling/grobid/$model/$model"-[0-9][0-9][0-9][0-9][0-9][0-9].train{,.gz}; do
        [[ -f "$file" ]] || continue
        date=${file##*/"$model"-}
        date=${date%%.*}
        if [[ -z "$latest" || "$date" > "$latest_date" ]]; then
            latest=$file
            latest_date=$date
        fi
    done
    printf '%s' "$latest"
}

# The file a group cannot train without, relative to the checkout.
group_data() {
    case "$1" in
        ner) echo data/sequenceLabelling/CoNLL-2003/eng.train ;;
        insult) echo data/sequenceLabelling/toxic/train.xml ;;
        dataset) echo data/sequenceLabelling/datasets/dataseer_sentences.json ;;
        citation) echo data/textClassification/citations/citation_sentiment_corpus.txt ;;
        dataseer) echo data/textClassification/dataseer/all-binary.csv ;;
        license) echo data/textClassification/licenses/copyrights-licenses-data-validated.csv ;;
        software) echo data/textClassification/software/software-use.json.gz ;;
        software-context) echo data/textClassification/software/software-contexts.json.gz ;;
        toxic) echo data/textClassification/toxic/train.csv ;;
    esac
}

group_module() {
    case "$1" in
        grobid) echo grobidTagger ;;
        ner) echo nerTagger ;;
        insult) echo insultTagger ;;
        dataset) echo datasetTagger ;;
        citation) echo citationClassifier ;;
        dataseer) echo dataseerClassifier ;;
        license) echo licenseClassifier ;;
        software) echo softwareClassifier ;;
        software-context) echo softwareContextClassifier ;;
        toxic) echo toxicCommentClassifier ;;
    esac
}

is_tagger() {
    contains "$1" grobid ner insult dataset
}

# The tasks of the run, one per line of TASKS: group, row (the GROBID model, or the group),
# architecture, embedding and transformer ('-' for none of the last two).
build_tasks() {
    local group architectures rows row architecture embedding transformer
    local row_index=0 static_index bert_index choices choice
    TASKS=()
    for group in "${RUN_GROUPS[@]}"; do
        if [[ "$group" == grobid ]]; then
            rows=("${MODELS[@]}")
            architectures=("${GROBID_ARCHITECTURES[@]}")
        else
            rows=("$group")
            if is_tagger "$group"; then
                # only the GROBID training files have the columns of the FEATURES architectures
                architectures=()
                for architecture in "${GROBID_ARCHITECTURES[@]}"; do
                    [[ "$architecture" == *FEATURES ]] || architectures+=("$architecture")
                done
            else
                architectures=()
                for architecture in "${CLASSIFIER_ARCHITECTURES[@]}"; do
                    # the toxic comment classifier is multi-label, which bert does not do
                    [[ "$group" == toxic && "$architecture" == bert ]] || architectures+=("$architecture")
                done
            fi
        fi
        for row in "${rows[@]}"; do
            static_index=0
            bert_index=0
            for architecture in "${architectures[@]}"; do
                if is_tagger "$group"; then
                    contains "$architecture" "${SEQUENCE_ARCHITECTURES[@]}" || continue
                fi
                if [[ "$architecture" == BERT* || "$architecture" == bert ]]; then
                    choices=("${TRANSFORMERS[@]}")
                    choice=$(((row_index + bert_index) % ${#choices[@]}))
                    bert_index=$((bert_index + 1))
                else
                    choices=("${EMBEDDINGS[@]}")
                    choice=$(((row_index + static_index) % ${#choices[@]}))
                    static_index=$((static_index + 1))
                fi
                [[ "$PRODUCT" != true ]] || choice=all
                local index
                for index in "${!choices[@]}"; do
                    [[ "$choice" == all || "$choice" == "$index" ]] || continue
                    embedding=-
                    transformer=-
                    if [[ "$architecture" == BERT* || "$architecture" == bert ]]; then
                        transformer=${choices[$index]}
                    else
                        embedding=${choices[$index]}
                    fi
                    TASKS+=("$group $row $architecture $embedding $transformer")
                done
            done
            row_index=$((row_index + 1))
        done
    done
}

# The command of a step of a task in the global CMD, empty when the application has no such
# step. The options of a training are given to the steps that train only.
build_step_command() {
    local step=$1 group=$2 row=$3 architecture=$4 embedding=$5 transformer=$6
    local module training=() action=""
    module=$(group_module "$group")
    CMD=()

    [[ "$embedding" == - || "$embedding" == none ]] || training+=(--embedding "$embedding")
    [[ "$transformer" == - ]] || training+=(--transformer "$transformer")
    [[ -z "$MAX_EPOCH" ]] || training+=(--max-epoch "$MAX_EPOCH")

    local base=("$PYTHON_BIN" -m "delft.applications.$module")
    case "$group" in
        grobid)
            case "$step" in
                train_eval | train) CMD=("${base[@]}" "$row" "$step" "${training[@]}") ;;
                nfold) CMD=("${base[@]}" "$row" train_eval "${training[@]}" --fold-count "$FOLD_COUNT") ;;
                eval) CMD=("${base[@]}" "$row" eval --input "$(latest_train_file "$row")") ;;
                tag)
                    # the models the tagger has sample texts for, which hold no features
                    if [[ "$architecture" != *FEATURES ]] &&
                        contains "$row" date citation name-citation name-header software; then
                        CMD=("${base[@]}" "$row" tag)
                    fi
                    ;;
            esac
            ;;
        ner)
            # tag needs a text file to annotate
            case "$step" in
                train_eval | train | eval) action=$step ;;
                nfold) action=train_eval ;;
            esac
            [[ -z "$action" ]] || CMD=("${base[@]}" "$action" --dataset-type conll2003)
            [[ "$step" != train_eval && "$step" != train && "$step" != nfold ]] || CMD+=("${training[@]}")
            [[ "$step" != nfold ]] || CMD+=(--fold-count "$FOLD_COUNT")
            ;;
        insult)
            case "$step" in
                train_eval | train) CMD=("${base[@]}" "$step" "${training[@]}") ;;
                nfold) CMD=("${base[@]}" train_eval "${training[@]}" --fold-count "$FOLD_COUNT") ;;
                tag) CMD=("${base[@]}" tag) ;;
            esac
            ;;
        dataset)
            # eval needs an evaluation file
            case "$step" in
                train_eval | train) CMD=("${base[@]}" "$step" "${training[@]}") ;;
                nfold) CMD=("${base[@]}" train_eval "${training[@]}" --fold-count "$FOLD_COUNT") ;;
                tag) CMD=("${base[@]}" tag) ;;
            esac
            ;;
        *)
            case "$step" in
                train_eval | train) CMD=("${base[@]}" "$step" "${training[@]}") ;;
                nfold) CMD=("${base[@]}" train_eval "${training[@]}" --fold-count "$FOLD_COUNT") ;;
                tag) CMD=("${base[@]}" classify) ;;
            esac
            ;;
    esac
    ((${#CMD[@]} == 0)) || CMD+=(--architecture "$architecture")
}

# The steps of a task, in the order they run: nfold last, so that eval and tag find the
# model of train rather than the models of the folds.
ordered_steps() {
    local step
    ORDERED_STEPS=()
    for step in train_eval train eval tag nfold; do
        ! contains "$step" "${STEPS[@]}" || ORDERED_STEPS+=("$step")
    done
}

# The working directory of a task: the data of the checkout linked, the models its own.
make_work_dir() {
    local work_dir=$1 entry
    mkdir -p "$work_dir/data/models"
    ln -sfn "$CONTAINER_WORKDIR/delft" "$work_dir/delft"
    for entry in "$CONTAINER_WORKDIR"/data/*; do
        case "${entry##*/}" in
            models | test-runs) ;;
            *) ln -sfn "$entry" "$work_dir/data/${entry##*/}" ;;
        esac
    done
}

run_task() {
    local index=$1 line group row architecture embedding transformer
    line=$(sed -n "$((index + 1))p" "$TASKS_FILE")
    [[ -n "$line" ]] || { echo "No task $index in $TASKS_FILE" >&2; exit 1; }
    read -r group row architecture embedding transformer <<<"$line"

    local name="$group $row $architecture"
    [[ "$embedding" == - ]] || name+=" embedding=$embedding"
    [[ "$transformer" == - ]] || name+=" transformer=$transformer"
    echo ">>> [task $index] START $name"

    # without the code of the checkout, python would run whatever delft its environment holds
    if [[ ! -d "$CONTAINER_WORKDIR/delft" || ! -d "$CONTAINER_WORKDIR/data" ]]; then
        echo ">>> [task $index] RESULT FAILED (no DeLFT checkout at $CONTAINER_WORKDIR) $name"
        exit 1
    fi
    echo ">>> [task $index] CHECKOUT $CONTAINER_WORKDIR $(git -C "$CONTAINER_WORKDIR" log -1 --format='%h %s' 2>/dev/null || true)"

    local work_dir="$WORK_ROOT/task_$index"
    make_work_dir "$work_dir"
    cd "$work_dir"
    export PYTHONPATH="$CONTAINER_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

    local step code start failed=0 trained=true
    ordered_steps
    for step in "${ORDERED_STEPS[@]}"; do
        build_step_command "$step" "$group" "$row" "$architecture" "$embedding" "$transformer"
        ((${#CMD[@]} > 0)) || continue
        if [[ "$trained" != true && ("$step" == eval || "$step" == tag) ]]; then
            echo ">>> [task $index] STEP $step SKIPPED (no trained model)"
            continue
        fi
        echo ">>> [task $index] STEP $step RUN ${CMD[*]}"
        start=$SECONDS
        code=0
        "${CMD[@]}" || code=$?
        if ((code == 0)); then
            echo ">>> [task $index] STEP $step OK $((SECONDS - start))s"
        else
            echo ">>> [task $index] STEP $step FAILED (exit $code) $((SECONDS - start))s"
            failed=$((failed + 1))
            [[ "$step" != train ]] || trained=false
        fi
    done

    cd "$CONTAINER_WORKDIR"
    if [[ "$KEEP_MODELS" != true && "$work_dir" == */task_"$index" ]]; then
        rm -rf -- "$work_dir"
    fi
    if ((failed > 0)); then
        echo ">>> [task $index] RESULT FAILED ($failed steps) $name"
        exit 1
    fi
    echo ">>> [task $index] RESULT PASSED $name"
}

# The outcome of every task of a run, from its logs.
report() {
    local log_dir=$1 tasks_file="$1/tasks.txt" total index line log passed=0 failed=0 pending=0
    [[ -f "$tasks_file" ]] || { echo "No tasks.txt in $log_dir" >&2; exit 1; }
    total=$(wc -l <"$tasks_file")
    for ((index = 0; index < total; index++)); do
        line=$(sed -n "$((index + 1))p" "$tasks_file")
        # the last log of the task: a task run again has several
        log=$(ls -t "$log_dir"/*_"$index".log 2>/dev/null | head -n 1 || true)
        if [[ -z "$log" ]]; then
            pending=$((pending + 1))
            printf '%4d  NOT RUN    %s\n' "$index" "$line"
        elif grep -q "^>>> \[task $index\] RESULT PASSED" "$log"; then
            passed=$((passed + 1))
        elif grep -q "^>>> \[task $index\] RESULT FAILED" "$log"; then
            failed=$((failed + 1))
            printf '%4d  FAILED     %s  (%s)\n' "$index" "$line" "$log"
            grep "^>>> \[task $index\] STEP .* FAILED" "$log" | sed 's/^>>> \[task [0-9]*\] /          /'
        else
            # running, or stopped before its end: out of time, out of memory, cancelled
            pending=$((pending + 1))
            printf '%4d  NO RESULT  %s  (%s)\n' "$index" "$line" "$log"
        fi
    done
    echo "$total tasks: $passed passed, $failed failed, $pending without a result"
    ((failed == 0 && pending == 0))
}

# An array task is started with `__task` and the settings the submitter resolved, as
# arguments: what a task runs does not depend on the environment reaching it.
if [[ "${1:-}" == __task ]]; then
    shift
    while (($# > 0)); do
        case "$1" in
            --tasks-file) TASKS_FILE=$2 ;;
            --steps) read -r -a STEPS <<<"$2" ;;
            --max-epoch) MAX_EPOCH=$2 ;;
            --fold-count) FOLD_COUNT=$2 ;;
            --work-root) WORK_ROOT=$2 ;;
            --keep-models) KEEP_MODELS=$2 ;;
            --python-bin) PYTHON_BIN=$2 ;;
            # sbatch runs a copy of this script from its spool directory: the checkout cannot
            # be told from where the script is, as it is on the login node
            --checkout) CONTAINER_WORKDIR=$2 ;;
            --index) TASK_INDEX=$2 ;;
            *) echo "Unknown task setting: $1" >&2; exit 2 ;;
        esac
        shift 2
    done
    run_task "${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?no array task index}}"
    exit 0
fi

PROFILE=${1:-}
[[ -n "$PROFILE" ]] || usage
shift
case "$PROFILE" in
    smoke)
        MAX_EPOCH=${MAX_EPOCH-3}
        FOLD_COUNT=${FOLD_COUNT:-2}
        ;;
    full)
        MAX_EPOCH=${MAX_EPOCH-}
        FOLD_COUNT=${FOLD_COUNT:-5}
        ;;
    report)
        [[ $# -eq 1 ]] || usage
        report "$1"
        exit
        ;;
    *) usage ;;
esac

cd "$CONTAINER_WORKDIR"
# the tasks run in directories of their own: the interpreter is given as an absolute path
[[ "$PYTHON_BIN" != */* || "$PYTHON_BIN" == /* ]] || PYTHON_BIN="$CONTAINER_WORKDIR/$PYTHON_BIN"

list_from_env "${STEPS:-}" "${ALL_STEPS[@]}"
STEPS=("${LIST[@]}")
for step in "${STEPS[@]}"; do
    contains "$step" "${ALL_STEPS[@]}" || { echo "Unknown step: $step" >&2; usage; }
done
list_from_env "${ARCHITECTURES:-}" "${GROBID_ARCHITECTURES[@]}"
SEQUENCE_ARCHITECTURES=("${LIST[@]}")
list_from_env "${CLASSIFIER_ARCHITECTURES:-}" "${DEFAULT_CLASSIFIER_ARCHITECTURES[@]}"
CLASSIFIER_ARCHITECTURES=("${LIST[@]}")
list_from_env "${EMBEDDINGS:-}" "${DEFAULT_EMBEDDINGS[@]}"
EMBEDDINGS=("${LIST[@]}")
list_from_env "${TRANSFORMERS:-}" "${DEFAULT_TRANSFORMERS[@]}"
TRANSFORMERS=("${LIST[@]}")

# the GROBID models: those asked for, or every one with a training file
if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<<"$MODELS"
else
    MODELS=()
    for directory in data/sequenceLabelling/grobid/*/; do
        model=$(basename "$directory")
        [[ -z "$(latest_train_file "$model")" ]] || MODELS+=("$model")
    done
fi

# the groups: those asked for, or all of them, without those whose data is missing
if (($# > 0)); then
    REQUESTED_GROUPS=("$@")
else
    REQUESTED_GROUPS=("${ALL_GROUPS[@]}")
fi
RUN_GROUPS=()
for group in "${REQUESTED_GROUPS[@]}"; do
    contains "$group" "${ALL_GROUPS[@]}" || { echo "Unknown group: $group" >&2; usage; }
    if [[ "$group" == grobid ]]; then
        for model in "${MODELS[@]}"; do
            [[ -n "$(latest_train_file "$model")" ]] || {
                echo "No training file for the GROBID model '$model'" >&2
                exit 1
            }
        done
        ((${#MODELS[@]} > 0)) || { echo "Skipping grobid: no training file" >&2; continue; }
    elif [[ ! -e "$(group_data "$group")" ]]; then
        echo "Skipping $group: $(group_data "$group") is missing" >&2
        continue
    fi
    RUN_GROUPS+=("$group")
done

build_tasks
TOTAL_TASKS=${#TASKS[@]}
((TOTAL_TASKS > 0)) || { echo "Nothing to run" >&2; exit 1; }
ARRAY_SPEC=${ARRAY_SPEC:-"0-$((TOTAL_TASKS - 1))"}
RUN_NAME="${PROFILE}_$(date +%Y%m%d_%H%M%S)"
WORK_ROOT=${WORK_ROOT:-"$CONTAINER_WORKDIR/data/test-runs/$RUN_NAME"}

if [[ "$DRY_RUN" == true ]]; then
    echo "Profile '$PROFILE': $TOTAL_TASKS tasks (array spec: $ARRAY_SPEC)"
    ordered_steps
    for ((i = 0; i < TOTAL_TASKS; i++)); do
        read -r group row architecture embedding transformer <<<"${TASKS[$i]}"
        printf '%3d  %s\n' "$i" "${TASKS[$i]}"
        for step in "${ORDERED_STEPS[@]}"; do
            build_step_command "$step" "$group" "$row" "$architecture" "$embedding" "$transformer"
            ((${#CMD[@]} == 0)) || printf '       %-10s %s\n' "$step" "${CMD[*]}"
        done
    done
    exit 0
fi

LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/test_${RUN_NAME}"}
mkdir -p "$LOG_DIR"
# what the tasks read their settings from, and what `report` reads the tasks from
TASKS_FILE="$LOG_DIR/tasks.txt"
printf '%s\n' "${TASKS[@]}" >"$TASKS_FILE"

SBATCH_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
    SBATCH_OPTS+=(--container-image="$CONTAINER_IMAGE"
                  --container-workdir="$CONTAINER_WORKDIR"
                  --container-mounts="$CONTAINER_MOUNTS")
fi
# shellcheck disable=SC2206  # SBATCH_EXTRA is a string of options to split on spaces
SBATCH_OPTS+=(--export=ALL
              --cpus-per-task="$CPUS_PER_TASK"
              --mem="$MEMORY"
              -p "$PARTITIONS"
              --gpus=1
              --nodes=1
              --time="$TIME_LIMIT"
              --job-name="delft_test_$PROFILE"
              --array="${ARRAY_SPEC}%${MAX_PARALLEL_JOBS}"
              --output="$LOG_DIR/%A_%a.log"
              --error="$LOG_DIR/%A_%a.log"
              $SBATCH_EXTRA)

TASK_ARGS=(--tasks-file "$TASKS_FILE" --steps "${STEPS[*]}" --max-epoch "$MAX_EPOCH"
           --fold-count "$FOLD_COUNT" --work-root "$WORK_ROOT" --keep-models "$KEEP_MODELS"
           --python-bin "$PYTHON_BIN" --checkout "$CONTAINER_WORKDIR")

job_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" "$(readlink -f "$0")" __task "${TASK_ARGS[@]}" | cut -d';' -f1)

echo "Submitted array $job_id: tasks $ARRAY_SPEC of $TOTAL_TASKS (max $MAX_PARALLEL_JOBS concurrent)"
echo "Logs:    $LOG_DIR"
echo "Monitor: squeue -j $job_id"
echo "Report:  $0 report $LOG_DIR"
