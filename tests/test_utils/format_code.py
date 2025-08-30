from pathlib import Path

def folder_to_prompt_string(folder_paths: list[Path], file_extensions: list[str]= ['.py', '.json', '.sql', '.db'], exclude_files: list[str]= ['__init__.py']) -> str:
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
                    # Limit to first 3000 lines to avoid filling context window
                    lines = file_content.split('\n')
                    if len(lines) > 3000:
                        truncated_content = '\n'.join(lines[:3000])
                        truncated_content += f"\n\n... (truncated: showing first 3000 of {len(lines)} lines)"
                    else:
                        truncated_content = file_content
                    
                    content.append(f"File Name: {relative_path}")
                    content.append('-----------------------------')
                    content.append(f"File Content: \n\n{truncated_content}\n\n")
                except UnicodeDecodeError:
                    print(f"Warning: Could not read file {file_path} as UTF-8")
                    continue

    return '\n\n'.join(content)

    
    