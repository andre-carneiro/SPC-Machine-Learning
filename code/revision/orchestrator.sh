#!/bin/zsh
# Waits for the replication/distribution queues, then runs the remaining experiments in three parallel streams.
# Logs in v2/logs/. Each script is resumable (checkpoints), so the orchestrator can be relaunched safely.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
echo "$(date) waiting for queues" >> logs/orchestrator.log
while pgrep -f "code/revision/run_queue.py" > /dev/null || pgrep -f "final_rule.py E4_mc_replicates/tri_rep1" > /dev/null; do sleep 120; done
echo "$(date) queues finished" >> logs/orchestrator.log
( $PY code/revision/final_rule.py E4_mc_replicates/tri_rep2 E4_mc_replicates/tri_rep3 E4_mc_replicates/tri_rep4 E4_mc_replicates/tri_rep5 E4_mc_replicates/tri_rep6 E5_distributions/gamma E5_distributions/pert > logs/final_rule_1.log 2>&1
  echo "$(date) final_rule stream 1 done" >> logs/orchestrator.log
  $PY code/revision/e8_thresholds.py > logs/e8.log 2>&1
  echo "$(date) e8 done" >> logs/orchestrator.log ) &
( $PY code/revision/final_rule.py E4_mc_replicates/tri_rep7 E4_mc_replicates/tri_rep8 E4_mc_replicates/tri_rep9 E4_mc_replicates/tri_rep10 E5_distributions/cmpoisson > logs/final_rule_2.log 2>&1
  echo "$(date) final_rule stream 2 done" >> logs/orchestrator.log
  $PY code/revision/e7_selection_methods.py > logs/e7.log 2>&1
  echo "$(date) e7 done" >> logs/orchestrator.log ) &
( $PY code/revision/e4b_e6_fixed_config.py e4b > logs/e4b.log 2>&1
  echo "$(date) e4b done" >> logs/orchestrator.log
  $PY code/revision/e4b_e6_fixed_config.py e6 > logs/e6.log 2>&1
  echo "$(date) e6 done" >> logs/orchestrator.log ) &
wait
echo "$(date) ALL DONE" >> logs/orchestrator.log
