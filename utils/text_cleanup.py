import re
import json

def remove_consecutive_duplicates(messages):
    """
    Removes back-to-back duplicate strings from a list of messages.
    """
    if not messages or len(messages) < 2:
        return messages

    cleaned = [messages[0]]
    for i in range(1, len(messages)):
        if messages[i] != messages[i - 1]:
            cleaned.append(messages[i])
    return cleaned

def strip_code_blocks(text):
    """
    Strips triple-backtick code blocks from GPT output.
    """
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()

def extract_json(text):
    """
    Attempts to extract a JSON object or array from GPT response text.
    """
    text = strip_code_blocks(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues (e.g. single quotes, trailing commas)
        try:
            fixed = (
                text.replace("'", '"')
                    .replace("\n", "")
                    .replace(",}", "}")
                    .replace(",]", "]")
            )
            return json.loads(fixed)
        except:
            return None
