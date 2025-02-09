from typing import List

def extract_code_blocks(content: str) -> List[str]:
    """Helper to extract code blocks from markdown content"""
    code_blocks = []
    parts = content.split("```python")
    for part in parts[1:]:  # Skip first part before any code block
        if "```" in part:
            code = part.split("```")[0].strip()
            code_blocks.append(code)
    return code_blocks