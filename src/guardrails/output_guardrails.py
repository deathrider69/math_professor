"""
clean_and_format:
input: str - raw math output string
output: str - cleaned and formatted math output string
"""

import re

class MathOutputGuardrails:
    def __init__(self):
        # Patterns to remove harmful scripts and tags
        self.harmful_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'import\s+os',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'delete\s+from',
            r'drop\s+table',
            r'union\s+select',
            r'<iframe[^>]*>.*?</iframe>',
            r'<embed[^>]*>.*?</embed>',
            r'<object[^>]*>.*?</object>'
        ]

        # Tags to strip completely (like <think> ... </think>)
        self.strip_tags = ['think']

    def filter_harmful_content(self, text: str) -> str:
        # Remove harmful patterns
        for pattern in self.harmful_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove custom tags like <think>...</think> completely but keep inner text
        for tag in self.strip_tags:
            # Replace opening tag
            text = re.sub(fr'<{tag}[^>]*>', '', text, flags=re.IGNORECASE)
            # Replace closing tag
            text = re.sub(fr'</{tag}>', '', text, flags=re.IGNORECASE)
        
        return text

    def clean_and_format(self, raw_output: str) -> str:
        """
        Main method: filters harmful content but leaves LaTeX and markdown intact for Streamlit.
        """
        cleaned = self.filter_harmful_content(raw_output)

        # Optional: normalize line endings and multiple blank lines for neatness
        cleaned = re.sub(r'\r\n?', '\n', cleaned)  # Normalize Windows line endings to \n
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # No more than 2 consecutive line breaks

        # Strip trailing and leading whitespace on each line
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))

        return cleaned

