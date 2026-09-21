"""RL-side reward adapters: calibration, reweighting, and episode rescoring.

Pure reward math and configuration are owned by ``eco_planner.reward``. These
adapters read and write ``RolloutEpisode`` / training TensorDict state and are
intentionally kept in RL.
"""
