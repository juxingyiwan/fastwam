# Cluster execution

Log in to the allocated compute node manually with an SSH command before running training or evaluation. Launch scripts run on the current node; do not embed node login, logout, or Slurm job/step cancellation in them.

Preserve the user's login sessions and allocations. Updating scripts does not authorize restarting or terminating running work.
