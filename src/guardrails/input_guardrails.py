"""
process_query():
input: str - raw user query
output: dict - contains:
    - original_query: str - the original query
    - sanitized_query: str - sanitized version of the query
    - should_process: bool - whether the query is valid for processing
    - error_message: str or None - error message if validation fails
"""

import re
import unicodedata
from typing import Dict, Any, List


class MathInputGuardrails:
    """Input guardrails component for math solving agent."""
    
    def __init__(self):
        # Math-related keywords and patterns
        self.math_keywords = {
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'sum', 'difference', 'product', 'quotient'],
            'algebra': ['solve', 'equation', 'variable', 'expression', 'polynomial', 'factor', 'expand'],
            'geometry': ['area', 'perimeter', 'volume', 'triangle', 'circle', 'rectangle', 'square', 'angle'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate', 'integrate', 'slope'],
            'statistics': ['mean', 'median', 'mode', 'variance', 'standard deviation', 'probability'],
            'trigonometry': ['sin', 'cos', 'tan', 'sine', 'cosine', 'tangent', 'radians', 'degrees']
        }
        
        self.symbol_map = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '±': '+-',
            '∞': 'infinity',
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral',
            '∂': 'partial',
            '√': 'sqrt',
            '∝': 'proportional',
            '≈': 'approximately',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '°': 'degrees',
            'π': 'pi',
            'θ': 'theta',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'λ': 'lambda',
            'μ': 'mu',
            'σ': 'sigma',
            'φ': 'phi',
            'ψ': 'psi',
            'ω': 'omega'
        }
        
        self.injection_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'import\s+os',
            r'__import__',
            r'globals\s*\(',
            r'locals\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
    
    def is_math_related(self, query: str) -> tuple[bool, str, float]:
        query_lower = query.lower()
        math_symbols = any(symbol in query for symbol in self.symbol_map)
        number_pattern = r'\d+\.?\d*'
        has_numbers = bool(re.search(number_pattern, query))
        operation_pattern = r'[+\-*/^=<>]'
        has_operations = bool(re.search(operation_pattern, query))
        
        category_scores = {}
        for category, keywords in self.math_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score / len(keywords)
        
        confidence = 0.0
        if math_symbols:
            confidence += 0.3
        if has_numbers:
            confidence += 0.2
        if has_operations:
            confidence += 0.2
        if category_scores:
            confidence += 0.3 * max(category_scores.values())
        
        best_category = max(category_scores, key=category_scores.get) if category_scores else 'general'
        is_math = confidence > 0.3
        
        return is_math, best_category, min(confidence, 1.0)
    
    def sanitize_query(self, query: str) -> str:
        sanitized = query
        for pattern in self.injection_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        for symbol, replacement in self.symbol_map.items():
            sanitized = sanitized.replace(symbol, replacement)
        sanitized = unicodedata.normalize('NFKD', sanitized)
        sanitized = ' '.join(sanitized.split())
        
        html_entities = {
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&times;': '*',
            '&divide;': '/',
            '&minus;': '-',
            '&plusmn;': '+-'
        }
        for entity, replacement in html_entities.items():
            sanitized = sanitized.replace(entity, replacement)
        
        return sanitized.strip()
    
    def detect_harmful_content(self, query: str) -> List[str]:
        harmful_indicators = []
        
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in self.injection_patterns):
            harmful_indicators.append('code_injection')
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
        if re.search(email_pattern, query):
            harmful_indicators.append('email_address')
        if re.search(phone_pattern, query):
            harmful_indicators.append('phone_number')
        
        profanity_pattern = r'\b(fuck|shit|damn|bitch|asshole)\b'
        if re.search(profanity_pattern, query, re.IGNORECASE):
            harmful_indicators.append('profanity')
        
        return harmful_indicators
    
    def process_query(self, query: str) -> Dict[str, Any]:
        is_math, _, confidence = self.is_math_related(query)
        harmful_content = self.detect_harmful_content(query)
        sanitized_query = self.sanitize_query(query)
        should_process = is_math and not harmful_content

        if not is_math:
            error_message = f"Query does not appear to be math-related (confidence: {confidence:.2f})"
        elif harmful_content:
            error_message = f"Query contains harmful content: {', '.join(harmful_content)}"
        else:
            error_message = None

        return {
            'original_query': query,
            'sanitized_query': sanitized_query,
            'should_process': should_process,
            'error_message': error_message
        }
