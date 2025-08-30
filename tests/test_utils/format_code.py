from pathlib import Path

def folder_to_prompt_string(folder_paths: list[Path], file_extensions: list[str]= ['.py', '.json'], exclude_files: list[str]= ['__init__.py']) -> str:
    """
    Convert a list of folder paths to a prompt string.
    """
    content = []
    for folder_path in folder_paths:
        for file_path in sorted(Path(folder_path).glob('*')):
            if file_path.suffix in file_extensions and file_path.name not in exclude_files:
                relative_path = file_path.relative_to(folder_path)
                try:
                    file_content = file_path.read_text(encoding='utf-8')
                    content.append(f"File Name: {relative_path}")
                    content.append('-----------------------------')
                    content.append(f"File Content: \n\n{file_content}\n\n")
                except UnicodeDecodeError:
                    print(f"Warning: Could not read file {file_path} as UTF-8")
                    continue

    return '\n\n'.join(content)

    
    