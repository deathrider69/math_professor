"""
solve_math_problem:
input: str - The math problem to solve
output: dict - contains
    - solution: str - Step-by-step solution
    - answer: str - Final numerical answer
    - calculated: bool - True if successfully calculated, False otherwise
"""

from openai import OpenAI
import os
import re
import json
from typing import Dict, Any

class MathProblemSolver:
    """
    A module for solving math problems using DeepSeek R1 model via NVIDIA NIM API.
    
    This class provides a clean interface for solving mathematical problems
    and returning structured results with solution steps and final answers.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the MathProblemSolver.
        
        Args:
            api_key (str, optional): NIM API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("NIM_API_KEY")
        if not self.api_key:
            raise ValueError("NIM_API_KEY must be provided either as parameter or environment variable")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    
    def solve_math_problem(self, query: str) -> Dict[str, Any]:
        """
        Solve a math problem and return structured results.
        
        Args:
            query (str): The math problem to solve
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - solution (str): Step-by-step solution
                - answer (str): Final numerical answer
                - calculated (bool): True if successfully calculated, False otherwise
        """
        try:
            # Enhanced prompt for better structured output
            enhanced_prompt = f"""
            Solve the following math problem step by step. Please provide:
            1. A clear step-by-step solution
            2. The final answer or expression as the answer (give a numerical value if there are any constants like pi, e, etc.)
            
            Problem: {query}
            
            Please format your response clearly with steps and highlight the final answer or expression.
            """
            
            # Get streaming response
            full_response = self._get_streaming_response(enhanced_prompt)
            
            if not full_response:
                return {
                    "solution": "No response received from the model",
                    "answer": "N/A",
                    "calculated": False
                }
            
            # Parse the response to extract solution and answer
            solution, answer, calculated = self._parse_response(full_response)
            
            return {
                "solution": solution,
                "answer": answer,
                "calculated": calculated
            }
            
        except Exception as e:
            return {
                "solution": f"Error occurred: {str(e)}",
                "answer": "N/A",
                "calculated": False
            }
    
    def _get_streaming_response(self, input_query: str) -> str:
        """
        Get streaming response from the model and return complete response.
        
        Args:
            input_query (str): The query to send to the model
            
        Returns:
            str: Complete response from the model
        """
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[
                    {
                        "role": "user",
                        "content": input_query
                    }
                ],
                temperature=0.6,
                stream=True,
                max_tokens=4096,
                top_p=0.7
            )
            
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            
            return full_response
            
        except Exception as e:
            print(f"Error in streaming response: {e}")
            return ""
    
    def _parse_response(self, response: str) -> tuple:
        """
        Parse the model response to extract solution steps and final answer.
        
        Args:
            response (str): Raw response from the model
            
        Returns:
            tuple: (solution, answer, calculated)
        """
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to find the final answer using various patterns
            answer_patterns = [
                r'final answer[:\s]*([+-]?\d*\.?\d+)',
                r'answer[:\s]*([+-]?\d*\.?\d+)',
                r'result[:\s]*([+-]?\d*\.?\d+)',
                r'solution[:\s]*([+-]?\d*\.?\d+)',
                r'=\s*([+-]?\d*\.?\d+)\s*$',
                r'([+-]?\d*\.?\d+)\s*$'
            ]
            
            answer = "N/A"
            calculated = False
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Get the last match (usually the final answer)
                    answer = matches[-1]
                    calculated = True
                    break
            
            # If no numerical answer found, try to extract from the end of response
            if not calculated:
                # Look for numbers at the end of the response
                numbers = re.findall(r'([+-]?\d*\.?\d+)', response)
                if numbers:
                    answer = numbers[-1]
                    calculated = True
            
            # The solution is the entire response
            solution = response
            
            return solution, answer, calculated
            
        except Exception as e:
            return f"Error parsing response: {str(e)}", "N/A", False
    
    def solve_batch(self, queries: list) -> list:
        """
        Solve multiple math problems in batch.
        
        Args:
            queries (list): List of math problems to solve
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.solve_math_problem(query)
            results.append(result)
        return results
