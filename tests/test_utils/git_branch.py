import pathlib
import subprocess

def get_git_branch():
    """Get the current git branch name"""
    try:
        # First, locate the current (nested) repo's root based on this file's location
        file_dir = pathlib.Path(__file__).resolve().parent
        nested_root_proc = subprocess.run(
            ['git', '-C', str(file_dir), 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=True
        )
        nested_root = pathlib.Path(nested_root_proc.stdout.strip())

        # Attempt to read the parent repo's branch (one directory above the nested root)
        parent_dir = nested_root.parent
        parent_branch_proc = subprocess.run(
            ['git', '-C', str(parent_dir), 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True
        )
        if parent_branch_proc.returncode == 0 and parent_branch_proc.stdout.strip():
            return parent_branch_proc.stdout.strip()

        # Fallback to current (nested) repo branch if parent isn't a git repo
        current_branch_proc = subprocess.run(
            ['git', '-C', str(file_dir), 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return current_branch_proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("Failed to get git branch name")