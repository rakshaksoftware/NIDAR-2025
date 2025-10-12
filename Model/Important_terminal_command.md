## Server commands
1. Tmux(keeps training process running even if you close your laptop)
   
| Action                                 | Command                       | Notes                                 |
| -------------------------------------- | ----------------------------- | ------------------------------------- |
| **Start a new session**                | `tmux new -s <name>`          | e.g. `tmux new -s yolo_train`         |
| **Detach from current session**        | `Ctrl + B`, then `D`          | Keeps processes running in background |
| **List all sessions**                  | `tmux ls`                     | See active tmux sessions              |
| **Reattach to a session**              | `tmux attach -t <name>`       | Resume your training/logs             |
| **Attach to last session**             | `tmux a`                      | Shortcut if only one session exists   |
| **Kill (end) a session**               | `tmux kill-session -t <name>` | Stops all processes in that session   |
| **Rename a session**                   | `Ctrl + B`, then `$`          | Rename current session                |
| **Create new session (in background)** | `tmux new -d -s <name>`       | Starts a detached session directly    |

