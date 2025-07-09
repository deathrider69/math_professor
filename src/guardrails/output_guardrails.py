"""
clean_and_format:
input: str - raw math output string
output: str - cleaned and formatted math output string
"""

import re

class MathOutputGuardrails:
    def __init__(self):
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
        self.strip_tags = ['think']

    def filter_harmful_content(self, text: str) -> str:
        for pattern in self.harmful_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE | re.DOTALL)

        
        for tag in self.strip_tags:
            text = re.sub(fr'<{tag}[^>]*>', '', text, flags=re.IGNORECASE)
            text = re.sub(fr'</{tag}>', '', text, flags=re.IGNORECASE)
        
        return text

    def clean_and_format(self, raw_output: str) -> str:
        """
        Main method: filters harmful content but leaves LaTeX and markdown intact for Streamlit.
        """
        cleaned = self.filter_harmful_content(raw_output)
        cleaned = re.sub(r'\r\n?', '\n', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))

        return cleaned

