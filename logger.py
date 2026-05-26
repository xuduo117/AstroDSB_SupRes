import os
import sys
import datetime

class Logger:
    def __init__(self, log_dir=None):
        """
        Custom logger that handles rank-based output.
        If log_dir is None, it implies the worker is not rank 0 and shouldn't log.
        """
        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = open(os.path.join(self.log_dir, "train_log.txt"), "a")
        else:
            self.log_file = None

    def info(self, message):
        """Standard log output for training progress."""
        if self.log_file is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{now}] INFO: {message}"
            
            # Print to console
            print(log_entry)
            
            # Write to file
            self.log_file.write(log_entry + "\n")
            self.log_file.flush()

    def warn(self, message):
        """Warning logs for non-critical issues."""
        if self.log_file is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{now}] WARN: {message}"
            print(f"\033[93m{log_entry}\033[0m") # Yellow text
            self.log_file.write(log_entry + "\n")
            self.log_file.flush()

    def close(self):
        """Gracefully close the log file at the end of training."""
        if self.log_file:
            self.log_file.close()