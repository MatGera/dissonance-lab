import re
import json
from pathlib import Path


def parse_tags(text: str, tag_name: str) -> str:
    """Extract content between XML tags.
    
    Args:
        text: Text containing XML tags
        tag_name: Name of tag to extract (without < >)
    
    Returns:
        Content between tags, or empty string if not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_list(text: str, prefix: str = "-") -> list[str]:
    """Parse bullet-pointed list.
    
    Args:
        text: Text containing bullet list
        prefix: Bullet prefix to strip (default: "-")
    
    Returns:
        List of parsed items
    """
    return [
        line.strip().lstrip(prefix).strip()
        for line in text.split("\n")
        if line.strip()
    ]


def load_txt(path: str | Path) -> str:
    """Load text file.
    
    Args:
        path: Path to text file
    
    Returns:
        File contents as string
    """
    with open(path) as f:
        return f.read()


def load_json(path: str | Path) -> dict:
    """Load JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON as dict
    """
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL file.
    
    Args:
        path: Path to JSONL file
    
    Returns:
        List of parsed JSON objects
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Save data as JSONL file.
    
    Creates parent directories if they don't exist.
    
    Args:
        data: List of dicts to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

