import os
import sys
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import dspy
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.guardrails.input_guardrails import MathInputGuardrails
from src.guardrails.output_guardrails import MathOutputGuardrails
from src.knowledge_base.agentic_rag_kb import AIMOKnowledgeBaseComponent
from src.web_search.math_web_search import MathWebSearchComponent
from src.solver.deepseek_solver import MathProblemSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolutionSource(Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    SOLVER = "solver"
    HUMAN_FEEDBACK = "human_feedback"

@dataclass
class MathSolution:
    """Data class to represent a math solution"""
    solution: str
    answer: str
    confidence: float
    source: SolutionSource
    reasoning: str
    requires_feedback: bool = False
    feedback_history: list = None
    
    def __post_init__(self):
        if self.feedback_history is None:
            self.feedback_history = []

class HumanFeedbackSignature(dspy.Signature):
    """DSPy signature for human feedback processing"""
    original_solution = dspy.InputField(desc="The original mathematical solution")
    human_feedback = dspy.InputField(desc="Human feedback on the solution")
    improved_solution = dspy.OutputField(desc="Improved solution based on feedback")
    confidence_score = dspy.OutputField(desc="Confidence score for the improved solution")

class FeedbackProcessor(dspy.Module):
    """DSPy module for processing human feedback"""
    
    def __init__(self):
        super().__init__()
        self.feedback_chain = dspy.ChainOfThought(HumanFeedbackSignature)
    
    def forward(self, original_solution: str, human_feedback: str) -> Dict[str, Any]:
        """Process human feedback and improve solution"""
        try:
            result = self.feedback_chain(
                original_solution=original_solution,
                human_feedback=human_feedback
            )
            
            return {
                "improved_solution": result.improved_solution,
                "confidence": float(result.confidence_score) if result.confidence_score.replace('.', '').isdigit() else 0.8,
                "processed": True
            }
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "improved_solution": original_solution,
                "confidence": 0.5,
                "processed": False
            }

class MathAgentOrchestrator:
    """Main orchestrator for the mathematical professor system"""
    
    def __init__(self):
        # Initialize components
        self.input_guardrails = MathInputGuardrails()
        self.output_guardrails = MathOutputGuardrails()
        self.knowledge_base = AIMOKnowledgeBaseComponent()
        self.web_search = MathWebSearchComponent()
        self.solver = MathProblemSolver()
        self.feedback_processor = FeedbackProcessor()
        
        # Initialize knowledge base
        self.knowledge_base.initialize_knowledge_base()
        
        # Solution cache for feedback processing
        self.solution_cache: Dict[str, MathSolution] = {}
        
        logger.info("MathAgentOrchestrator initialized successfully")
    
    def _generate_solution_id(self, query: str) -> str:
        """Generate unique ID for solution tracking"""
        return f"sol_{hash(query + str(datetime.now()))}"
    
    def _check_knowledge_base(self, query: str) -> Tuple[bool, Optional[MathSolution]]:
        """Check knowledge base for existing solution"""
        try:
            kb_result = self.knowledge_base.query(query)
            
            if kb_result.get('is_present', False):
                solution = MathSolution(
                    solution=kb_result.get('solution', ''),
                    answer=kb_result.get('answer', ''),
                    confidence=0.9,
                    source=SolutionSource.KNOWLEDGE_BASE,
                    reasoning="Found in knowledge base"
                )
                return True, solution
            
            return False, None
            
        except Exception as e:
            logger.error(f"Knowledge base query error: {str(e)}")
            return False, None
    
    def _perform_web_search(self, query: str) -> Optional[MathSolution]:
        """Perform web search for solution"""
        try:
            search_result = self.web_search.search_math_query(query)
            
            if search_result.get('is_found', False):
                solution = MathSolution(
                    solution=search_result.get('solution', ''),
                    answer=str(search_result.get('answer', '')),
                    confidence=0.7,
                    source=SolutionSource.WEB_SEARCH,
                    reasoning="Found through web search"
                )
                return solution
            
            return None
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return None
    
    def _use_solver(self, query: str) -> Optional[MathSolution]:
        """Use internal solver as fallback"""
        try:
            solver_result = self.solver.solve_math_problem(query)
            
            if solver_result.get('calculated', False):
                solution = MathSolution(
                    solution=solver_result.get('solution', ''),
                    answer=solver_result.get('answer', ''),
                    confidence=0.8,
                    source=SolutionSource.SOLVER,
                    reasoning="Generated by internal solver"
                )
                return solution
            
            return None
            
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            return None
    
    def _should_request_feedback(self, solution: MathSolution) -> bool:
        """Determine if human feedback is needed"""
        # Request feedback for low confidence solutions or complex problems
        if solution.confidence < 0.6:
            return True
        
        # Check for complex mathematical concepts that benefit from human validation
        complex_keywords = [
            'theorem', 'proof', 'integral', 'derivative', 'limit',
            'differential', 'topology', 'abstract', 'group theory'
        ]
        
        if any(keyword in solution.solution.lower() for keyword in complex_keywords):
            return True
        
        return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        # Step 1: Input validation through guardrails
        guardrail_result = self.input_guardrails.process_query(query)
        
        if not guardrail_result.get('should_process', False):
            return {
                'success': False,
                'error': guardrail_result.get('error_message', 'Invalid query'),
                'solution_id': None
            }
        
        sanitized_query = guardrail_result.get('sanitized_query', query)
        solution_id = self._generate_solution_id(sanitized_query)
        
        # Step 2: Check knowledge base
        kb_found, kb_solution = self._check_knowledge_base(sanitized_query)
        
        if kb_found and kb_solution:
            final_solution = kb_solution
        else:
            # Step 3: Web search
            web_solution = self._perform_web_search(sanitized_query)
            
            if web_solution:
                final_solution = web_solution
            else:
                # Step 4: Use internal solver
                solver_solution = self._use_solver(sanitized_query)
                
                if solver_solution:
                    final_solution = solver_solution
                else:
                    return {
                        'success': False,
                        'error': 'Unable to find or generate solution',
                        'solution_id': solution_id
                    }
        
        # Step 5: Determine if feedback is needed
        final_solution.requires_feedback = self._should_request_feedback(final_solution)
        
        # Step 6: Apply output guardrails
        formatted_solution = self.output_guardrails.clean_and_format(final_solution.solution)
        final_solution.solution = formatted_solution
        
        # Cache solution for potential feedback
        self.solution_cache[solution_id] = final_solution
        
        return {
            'success': True,
            'solution_id': solution_id,
            'solution': final_solution.solution,
            'answer': final_solution.answer,
            'confidence': final_solution.confidence,
            'source': final_solution.source.value,
            'reasoning': final_solution.reasoning,
            'requires_feedback': final_solution.requires_feedback
        }
    
    def submit_feedback(self, solution_id: str, feedback: str) -> Dict[str, Any]:
        """Process human feedback for a solution"""
        if solution_id not in self.solution_cache:
            return {
                'success': False,
                'error': 'Solution not found in cache'
            }
        
        original_solution = self.solution_cache[solution_id]
        
        # Process feedback using DSPy
        feedback_result = self.feedback_processor.forward(
            original_solution.solution,
            feedback
        )
        
        if feedback_result.get('processed', False):
            # Create improved solution
            improved_solution = MathSolution(
                solution=feedback_result['improved_solution'],
                answer=original_solution.answer,
                confidence=feedback_result['confidence'],
                source=SolutionSource.HUMAN_FEEDBACK,
                reasoning=f"Improved based on feedback: {feedback}",
                requires_feedback=False,
                feedback_history=original_solution.feedback_history + [feedback]
            )
            
            # Update cache
            self.solution_cache[solution_id] = improved_solution
            
            return {
                'success': True,
                'improved_solution': improved_solution.solution,
                'confidence': improved_solution.confidence,
                'feedback_incorporated': True
            }
        
        return {
            'success': False,
            'error': 'Failed to process feedback'
        }
    
    def get_solution_history(self, solution_id: str) -> Dict[str, Any]:
        """Get solution history including feedback"""
        if solution_id not in self.solution_cache:
            return {
                'success': False,
                'error': 'Solution not found'
            }
        
        solution = self.solution_cache[solution_id]
        
        return {
            'success': True,
            'solution': solution.solution,
            'answer': solution.answer,
            'confidence': solution.confidence,
            'source': solution.source.value,
            'feedback_history': solution.feedback_history,
            'requires_feedback': solution.requires_feedback
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = MathAgentOrchestrator()
    
    # Test query
    test_query = "Solve the quadratic equation x^2 + 5x + 6 = 0"
    
    # Process query
    result = orchestrator.process_query(test_query)
    print("Processing Result:", result)
    
    # Test feedback if needed
    if result.get('requires_feedback', False):
        feedback = "Please provide more detailed explanation of the factoring step"
        feedback_result = orchestrator.submit_feedback(
            result['solution_id'], 
            feedback
        )
        print("Feedback Result:", feedback_result)