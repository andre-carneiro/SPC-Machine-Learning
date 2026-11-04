#!/bin/zsh
# Relaunch (01/11 ~10h) of the final-rule streams with the reference-family benchmark, then E7 and E8.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
echo "$(date) orchestrator_2 started" >> logs/orchestrator.log
( $PY code/revision/final_rule.py E4_mc_replicates/tri_rep2 E4_mc_replicates/tri_rep3 E4_mc_replicates/tri_rep4 E4_mc_replicates/tri_rep5 E4_mc_replicates/tri_rep6 E5_distributions/gamma E5_distributions/pert >> logs/final_rule_1.log 2>&1
  echo "$(date) final_rule stream 1 done" >> logs/orchestrator.log
  $PY code/revision/e8_thresholds.py > logs/e8.log 2>&1
  echo "$(date) e8 done" >> logs/orchestrator.log ) &
( $PY code/revision/final_rule.py E4_mc_replicates/tri_rep7 E4_mc_replicates/tri_rep8 E4_mc_replicates/tri_rep9 E4_mc_replicates/tri_rep10 E5_distributions/cmpoisson >> logs/final_rule_2.log 2>&1
  echo "$(date) final_rule stream 2 done" >> logs/orchestrator.log
  $PY code/revision/e7_selection_methods.py > logs/e7.log 2>&1
  echo "$(date) e7 done" >> logs/orchestrator.log ) &
wait
echo "$(date) orchestrator_2 ALL DONE" >> logs/orchestrator.log
