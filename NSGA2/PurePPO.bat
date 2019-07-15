REM Using Baldwinian mode with num_updates so high that first genome never finishes should be equivalent to pure PPO
REM %* Allows for any additional parameters
python NSGAII.py --evol-mode baldwin --num-gens 1 --pop-size 1 --use-proper-time-limits --watch-frequency 1 --num-processes 1 --num-updates 9765 %*