"""
search_math_query():
input: str - Mathematical query string
output: dict - contains
    - solution: str - Step-by-step solution
    - answer: float - Numeric answer
    - is_found: bool - Whether a solution was found
"""

import os
from tavily import TavilyClient
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from pathlib import Path

# Get path to the .env file in the parentdirectory
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

@dataclass
class MathSearchResult:
    """Data structure for math search results"""
    solution: Optional[str]
    answer: Optional[float]
    is_found: bool

class MathWebSearchComponent:
    """
    Web search component for mathematical queries using Tavily API
    Designed for integration into larger math agent systems
    """
    
    def __init__(self):
        """
        Initialize the math web search component
        
        Args:
            api_key (str): Tavily API key
        """
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def search_math_query(self, query: str) -> MathSearchResult:
        """
        Search for mathematical solutions using Tavily API
        
        Args:
            query (str): Mathematical query string
            
        Returns:
            MathSearchResult: Contains solution, answer, and is_found status
        """
        try:
            # Enhance query for better math results
            enhanced_query = self._enhance_math_query(query)
            
            # Perform the search
            search_results = self._perform_search(enhanced_query)
            
            if not search_results:
                return MathSearchResult(solution=None, answer=None, is_found=False)
            
            # Extract mathematical content from results
            math_content = self._extract_math_content(search_results)
            
            if not math_content:
                return MathSearchResult(solution=None, answer=None, is_found=False)
            
            # Parse solution and answer
            solution = self._extract_solution_steps(math_content)
            answer = self._extract_numeric_answer(math_content)
            
            if solution is not None and answer is not None:
                return MathSearchResult(solution=solution, answer=answer, is_found=True)
            else:
                return MathSearchResult(solution=None, answer=None, is_found=False)
                
        except Exception as e:
            print(f"Error in math search: {str(e)}")
            return MathSearchResult(solution=None, answer=None, is_found=False)
    
    def _enhance_math_query(self, query: str) -> str:
        """
        Enhance the query to get better mathematical results
        
        Args:
            query (str): Original query
            
        Returns:
            str: Enhanced query
        """
        # Add math-specific keywords to improve search results
        math_keywords = ["step by step", "solution", "solve", "mathematics", "calculation"]
        
        # Check if query already contains math keywords
        has_math_keywords = any(keyword in query.lower() for keyword in math_keywords)
        
        if not has_math_keywords:
            enhanced_query = f"{query} step by step solution mathematics"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _perform_search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Perform the actual search using Tavily Client with improved error handling
        
        Args:
            query (str): Search query
            
        Returns:
            Optional[Dict]: Search results or None if failed
        """
        try:
            # Check if Tavily API key is available
            if not os.getenv("TAVILY_API_KEY"):
                print("Tavily API key not found. Web search unavailable.")
                return None
            
            response = self.client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                max_results=5,
                include_domains=[
                    "wolfram.com",
                    "symbolab.com", 
                    "mathway.com",
                    "khanacademy.org",
                    "brilliant.org",
                    "stackexchange.com",
                    "math.stackexchange.com"
                ]
            )
            
            return response
                
        except Exception as e:
            error_msg = str(e)
            if "502" in error_msg:
                print(f"Tavily API temporarily unavailable (502 Bad Gateway). Web search will be skipped.")
            elif "404" in error_msg:
                print(f"Tavily API endpoint not found (404). Please check API configuration.")
            elif "401" in error_msg or "403" in error_msg:
                print(f"Tavily API authentication failed. Please check your API key.")
            else:
                print(f"Search request failed: {error_msg}")
            
            return None
    
    def _extract_math_content(self, search_results: Dict[str, Any]) -> str:
        """
        Extract mathematical content from search results, with score filtering

        Args:
            search_results (Dict): Raw search results

        Returns:
            str: Combined mathematical content
        """
        content_parts = []
        
        #print(search_results)

        # Extract from answer if available
        if "answer" in search_results and search_results["answer"]:
            content_parts.append(search_results["answer"])

        # Extract from results, filtering by score
        if "results" in search_results:
            for result in search_results["results"]:
                score = result.get("score", 0)  # Default to 0 if missing
                if score < 0.9:
                    continue
                print(f"Score: {score}")
                print(f"Result: {result}")
                if "content" in result and result["content"]:
                    print(score, result["content"])
                    content_parts.append(result["content"])
                if "raw_content" in result and result["raw_content"]:
                    print(score, result["content"])
                    content_parts.append(result["raw_content"])

        # Filter out None before joining
        content_parts = [part for part in content_parts if part is not None]

        return " ".join(content_parts)
    
    def _extract_solution_steps(self, content: str) -> Optional[str]:
        """
        Extract step-by-step solution from content
        
        Args:
            content (str): Raw content
            
        Returns:
            Optional[str]: Formatted solution steps
        """
        if not content:
            return None
        
        # Common patterns for step-by-step solutions
        step_patterns = [
            r'Step \d+:.*?(?=Step \d+:|$)',
            r'\d+\.\s.*?(?=\d+\.\s|$)',
            r'First,.*?(?=Second,|Next,|Then,|Finally,|$)',
            r'Solution:.*?(?=Answer:|Final Answer:|$)'
        ]
        
        solution_steps = []
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                solution_steps.extend(matches)
                break
        
        if solution_steps:
            # Clean and format the steps
            formatted_steps = []
            for step in solution_steps:
                cleaned_step = re.sub(r'\s+', ' ', step.strip())
                if cleaned_step and len(cleaned_step) > 10:  # Filter out very short matches
                    formatted_steps.append(cleaned_step)
            
            return "\n".join(formatted_steps) if formatted_steps else None
        
        # If no structured steps found, look for any mathematical explanation
        math_explanation_patterns = [
            r'To solve.*?(?=\n\n|\.|$)',
            r'We need to.*?(?=\n\n|\.|$)',
            r'Calculate.*?(?=\n\n|\.|$)'
        ]
        
        for pattern in math_explanation_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _extract_numeric_answer(self, content: str) -> Optional[float]:
        """
        Extract numeric answer from content
        
        Args:
            content (str): Raw content
            
        Returns:
            Optional[float]: Numeric answer
        """
        if not content:
            return None
        
        # Patterns to find answers
        answer_patterns = [
            r'(?:answer|result|solution)(?:\s*is|\s*=|\s*:)\s*([+-]?\d*\.?\d+)',
            r'(?:final answer|the answer)(?:\s*is|\s*=|\s*:)\s*([+-]?\d*\.?\d+)',
            r'(?:equals?|=)\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)\s*(?:is the answer|is the result)',
            r'x\s*=\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)$'  # Number at end of line
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    # Try to convert the first match to float
                    answer = float(matches[0])
                    return answer
                except (ValueError, TypeError):
                    continue
        
        # Look for fractions and convert them
        fraction_pattern = r'(\d+)/(\d+)'
        fraction_matches = re.findall(fraction_pattern, content)
        if fraction_matches:
            try:
                numerator, denominator = fraction_matches[0]
                return float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                pass
        
        return None
    
    def batch_search(self, queries: list) -> Dict[str, MathSearchResult]:
        """
        Search multiple math queries in batch
        
        Args:
            queries (list): List of mathematical queries
            
        Returns:
            Dict[str, MathSearchResult]: Results for each query
        """
        results = {}
        for query in queries:
            results[query] = self.search_math_query(query)
        return results
