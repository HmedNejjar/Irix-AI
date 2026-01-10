import re, json

def extract_json_robust(text):
    """
    Most comprehensive JSON extraction
    """
    text = text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        '```json',
        '```',
        'JSON:',
        'Response:',
        'Answer:'
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Method 1: Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Find JSON object
    start = text.find('{')
    if start != -1:
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break
    
    # Method 3: Find JSON array
    start = text.find('[')
    if start != -1:
        bracket_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break
    
    # Method 4: Use regex for code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None