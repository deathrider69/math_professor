import os
import sys
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
import dspy
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

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

class MathAgentState(TypedDict):
    """State for LangGraph math agent workflow"""
    original_query: str
    sanitized_query: str
    solution: str
    answer: str
    confidence: float
    source: str
    reasoning: str
    requires_feedback: bool
    feedback_history: List[str]
    current_step: str
    error_message: Optional[str]
    processing_complete: bool

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

# DSPy Signatures for feedback processing
class SolutionRefinementSignature(dspy.Signature):
    """DSPy signature for refining mathematical solutions based on human feedback"""
    original_problem = dspy.InputField(desc="The original mathematical problem")
    original_solution = dspy.InputField(desc="The original solution provided")
    human_feedback = dspy.InputField(desc="Human feedback on the solution")
    refined_solution = dspy.OutputField(desc="Refined solution addressing the feedback")
    confidence_score = dspy.OutputField(desc="Confidence score for the refined solution (0-1)")

class FeedbackProcessor:
    """DSPy-based feedback processor using Gemini"""
    
    def __init__(self, gemini_api_key: str):
        # Configure DSPy with Gemini using LiteLLM
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        
        # Use LiteLLM integration for Gemini
        self.lm = dspy.LM('gemini/gemini-2.0-flash')
        dspy.configure(lm=self.lm)
        
        # Initialize DSPy modules
        self.refine_solution = dspy.ChainOfThought(SolutionRefinementSignature)
        self.feedback_history = []
        
        logger.info("DSPy Feedback Processor initialized with Gemini")
    
    def process_feedback(self, original_problem: str, original_solution: str, 
                        human_feedback: str) -> Dict[str, Any]:
        """Process human feedback and refine solution"""
        try:
            result = self.refine_solution(
                original_problem=original_problem,
                original_solution=original_solution,
                human_feedback=human_feedback
            )
            
            # Store feedback
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "original_solution": original_solution,
                "feedback": human_feedback,
                "refined_solution": result.refined_solution,
                "confidence": float(result.confidence_score) if result.confidence_score.replace('.', '').isdigit() else 0.8
            }
            self.feedback_history.append(feedback_entry)
            
            return {
                "success": True,
                "refined_solution": result.refined_solution,
                "confidence": feedback_entry["confidence"],
                "feedback_incorporated": True
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "refined_solution": original_solution,
                "confidence": 0.5
            }

class MathAgentOrchestrator:
    """LangGraph-based orchestrator for the mathematical professor system"""
    
    def __init__(self, gemini_api_key: str = None):
        # Initialize components
        self.input_guardrails = MathInputGuardrails()
        self.output_guardrails = MathOutputGuardrails()
        self.knowledge_base = AIMOKnowledgeBaseComponent()
        self.web_search = MathWebSearchComponent()
        self.solver = MathProblemSolver()
        
        # Initialize DSPy feedback processor
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            self.feedback_processor = FeedbackProcessor(self.gemini_api_key)
        else:
            logger.warning("No Gemini API key provided. Feedback functionality will be limited.")
            self.feedback_processor = None
        
        # Initialize knowledge base
        try:
            self.knowledge_base.initialize_knowledge_base()
        except Exception as e:
            logger.warning(f"Knowledge base initialization failed: {e}")
        
        # Solution cache for feedback processing
        self.solution_cache: Dict[str, MathSolution] = {}
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("LangGraph MathAgentOrchestrator initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(MathAgentState)
        
        # Add nodes
        workflow.add_node("input_validation", self._validate_input)
        workflow.add_node("knowledge_search", self._search_knowledge_base)
        workflow.add_node("web_search", self._perform_web_search)
        workflow.add_node("ai_solver", self._use_ai_solver)
        workflow.add_node("output_formatting", self._format_output)
        workflow.add_node("feedback_processing", self._process_feedback)
        
        # Define the workflow edges
        workflow.set_entry_point("input_validation")
        
        workflow.add_conditional_edges(
            "input_validation",
            self._should_continue_after_validation,
            {
                "continue": "knowledge_search",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "knowledge_search",
            self._should_continue_after_kb,
            {
                "found": "output_formatting",
                "not_found": "web_search"
            }
        )
        
        workflow.add_conditional_edges(
            "web_search",
            self._should_continue_after_web,
            {
                "found": "output_formatting",
                "not_found": "ai_solver"
            }
        )
        
        workflow.add_edge("ai_solver", "output_formatting")
        workflow.add_edge("output_formatting", END)
        workflow.add_edge("feedback_processing", "output_formatting")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _validate_input(self, state: MathAgentState) -> MathAgentState:
        """Validate and sanitize input using guardrails"""
        try:
            result = self.input_guardrails.process_query(state["original_query"])
            
            if result.get("should_process", False):
                state["sanitized_query"] = result.get("sanitized_query", state["original_query"])
                state["current_step"] = "input_validated"
                state["error_message"] = None
            else:
                state["error_message"] = result.get("error_message", "Invalid query")
                state["processing_complete"] = True
                
        except Exception as e:
            state["error_message"] = f"Input validation error: {str(e)}"
            state["processing_complete"] = True
            
        return state
    
    def _search_knowledge_base(self, state: MathAgentState) -> MathAgentState:
        """Search the knowledge base for existing solutions"""
        try:
            kb_result = self.knowledge_base.query(state["sanitized_query"])
            
            if kb_result.get("is_present", False):
                state["solution"] = kb_result.get("solution", "")
                state["answer"] = kb_result.get("answer", "")
                state["confidence"] = 0.9
                state["source"] = SolutionSource.KNOWLEDGE_BASE.value
                state["reasoning"] = "Found in knowledge base"
                state["current_step"] = "knowledge_found"
            else:
                state["current_step"] = "knowledge_not_found"
                
        except Exception as e:
            logger.error(f"Knowledge base search error: {str(e)}")
            state["current_step"] = "knowledge_error"
            
        return state
    
    def _perform_web_search(self, state: MathAgentState) -> MathAgentState:
        """Perform web search for solutions with improved error handling"""
        try:
            search_result = self.web_search.search_math_query(state["sanitized_query"])
            
            if search_result.is_found:
                state["solution"] = search_result.solution or ""
                state["answer"] = str(search_result.answer) if search_result.answer else ""
                state["confidence"] = 0.7
                state["source"] = SolutionSource.WEB_SEARCH.value
                state["reasoning"] = "Found through web search"
                state["current_step"] = "web_found"
            else:
                state["current_step"] = "web_not_found"
                logger.info("Web search did not find a solution, proceeding to AI solver")
                
        except Exception as e:
            logger.warning(f"Web search error (will continue with AI solver): {str(e)}")
            state["current_step"] = "web_not_found"  # Continue to AI solver instead of failing
            
        return state
    
    def _use_ai_solver(self, state: MathAgentState) -> MathAgentState:
        """Use AI solver as fallback"""
        try:
            solver_result = self.solver.solve_math_problem(state["sanitized_query"])
            
            if solver_result.get("calculated", False):
                state["solution"] = solver_result.get("solution", "")
                state["answer"] = solver_result.get("answer", "")
                state["confidence"] = 0.8
                state["source"] = SolutionSource.SOLVER.value
                state["reasoning"] = "Generated by AI solver"
                state["current_step"] = "solver_completed"
            else:
                state["error_message"] = "Unable to solve using any method"
                state["current_step"] = "solver_failed"
                
        except Exception as e:
            logger.error(f"AI solver error: {str(e)}")
            state["error_message"] = f"AI solver error: {str(e)}"
            state["current_step"] = "solver_error"
            
        return state
    
    def _format_output(self, state: MathAgentState) -> MathAgentState:
        """Format output using guardrails"""
        try:
            if state.get("solution"):
                formatted_solution = self.output_guardrails.clean_and_format(state["solution"])
                state["solution"] = formatted_solution
                
                # Determine if feedback is needed
                state["requires_feedback"] = self._should_request_feedback(state)
                
            state["processing_complete"] = True
            state["current_step"] = "completed"
            
        except Exception as e:
            logger.error(f"Output formatting error: {str(e)}")
            state["error_message"] = f"Output formatting error: {str(e)}"
            
        return state
    
    def _process_feedback(self, state: MathAgentState) -> MathAgentState:
        """Process human feedback to refine solution"""
        if not self.feedback_processor:
            state["error_message"] = "Feedback processor not available"
            return state
            
        try:
            # This will be called separately for feedback processing
            state["source"] = SolutionSource.HUMAN_FEEDBACK.value
            state["current_step"] = "feedback_processed"
            
        except Exception as e:
            logger.error(f"Feedback processing error: {str(e)}")
            state["error_message"] = f"Feedback processing error: {str(e)}"
            
        return state
    
    # Conditional edge functions
    def _should_continue_after_validation(self, state: MathAgentState) -> str:
        if state.get("error_message"):
            return "end"
        return "continue"
    
    def _should_continue_after_kb(self, state: MathAgentState) -> str:
        if state["current_step"] == "knowledge_found":
            return "found"
        return "not_found"
    
    def _should_continue_after_web(self, state: MathAgentState) -> str:
        if state["current_step"] == "web_found":
            return "found"
        return "not_found"
    
    def _should_request_feedback(self, state: MathAgentState) -> bool:
        """Determine if human feedback is needed"""
        confidence = state.get("confidence", 0)
        if confidence < 0.6:
            return True
        
        # Check for complex mathematical concepts
        solution = state.get("solution", "").lower()
        complex_keywords = [
            'theorem', 'proof', 'integral', 'derivative', 'limit',
            'differential', 'topology', 'abstract', 'group theory'
        ]
        
        if any(keyword in solution for keyword in complex_keywords):
            return True
        
        return False
    
    def _generate_solution_id(self, query: str) -> str:
        """Generate unique ID for solution tracking"""
        return f"sol_{hash(query + str(datetime.now()))}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main processing pipeline using LangGraph"""
        solution_id = self._generate_solution_id(query)
        
        # Initialize state
        initial_state = MathAgentState(
            original_query=query,
            sanitized_query="",
            solution="",
            answer="",
            confidence=0.0,
            source="",
            reasoning="",
            requires_feedback=False,
            feedback_history=[],
            current_step="starting",
            error_message=None,
            processing_complete=False
        )
        
        try:
            # Run the LangGraph workflow
            final_state = self.workflow.invoke(initial_state, config={"configurable": {"thread_id": solution_id}})
            
            if final_state.get("error_message"):
                return {
                    'success': False,
                    'error': final_state["error_message"],
                    'solution_id': solution_id,
                    'stage': final_state.get("current_step", "unknown")
                }
            
            # Create solution object for caching
            solution = MathSolution(
                solution=final_state.get("solution", ""),
                answer=final_state.get("answer", ""),
                confidence=final_state.get("confidence", 0.0),
                source=SolutionSource(final_state.get("source", "solver")),
                reasoning=final_state.get("reasoning", ""),
                requires_feedback=final_state.get("requires_feedback", False)
            )
            
            # Cache solution for potential feedback
            self.solution_cache[solution_id] = solution
            
            return {
                'success': True,
                'solution_id': solution_id,
                'solution': solution.solution,
                'answer': solution.answer,
                'confidence': solution.confidence,
                'source': solution.source.value,
                'reasoning': solution.reasoning,
                'requires_feedback': solution.requires_feedback,
                'current_step': final_state.get("current_step", "completed")
            }
            
        except Exception as e:
            logger.error(f"LangGraph workflow error: {str(e)}")
            return {
                'success': False,
                'error': f"Workflow execution error: {str(e)}",
                'solution_id': solution_id,
                'stage': "workflow_error"
            }
    
    def submit_feedback(self, solution_id: str, feedback: str) -> Dict[str, Any]:
        """Process human feedback for a solution using DSPy"""
        if solution_id not in self.solution_cache:
            return {
                'success': False,
                'error': 'Solution not found in cache'
            }
        
        if not self.feedback_processor:
            return {
                'success': False,
                'error': 'Feedback processor not available. Please provide Gemini API key.'
            }
        
        original_solution = self.solution_cache[solution_id]
        
        # Process feedback using DSPy
        feedback_result = self.feedback_processor.process_feedback(
            original_solution.solution,
            original_solution.solution,
            feedback
        )
        
        if feedback_result.get('success', False):
            # Create improved solution
            improved_solution = MathSolution(
                solution=feedback_result['refined_solution'],
                answer=original_solution.answer,
                confidence=feedback_result['confidence'],
                source=SolutionSource.HUMAN_FEEDBACK,
                reasoning=f"Improved based on feedback: {feedback}",
                requires_feedback=False,
                feedback_history=original_solution.feedback_history + [feedback]
            )
            
            # Update cache
            self.solution_cache[solution_id] = improved_solution
            
            # Format the improved solution
            formatted_solution = self.output_guardrails.clean_and_format(improved_solution.solution)
            improved_solution.solution = formatted_solution
            
            return {
                'success': True,
                'improved_solution': improved_solution.solution,
                'confidence': improved_solution.confidence,
                'feedback_incorporated': True,
                'source': 'human_feedback'
            }
        
        return {
            'success': False,
            'error': feedback_result.get('error', 'Failed to process feedback')
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
    
    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """Get feedback history from DSPy processor"""
        if not self.feedback_processor:
            return []
        
        return self.feedback_processor.feedback_history
    
    def health_check(self) -> Dict[str, str]:
        """Check system health status"""
        status = {}
        
        # Check core components
        try:
            # Test input guardrails
            test_result = self.input_guardrails.process_query("test query")
            status["guardrails"] = "healthy" if test_result else "error"
        except Exception:
            status["guardrails"] = "error"
        
        try:
            # Test knowledge base
            kb_status = self.knowledge_base.get_status()
            status["knowledge_base"] = "healthy" if kb_status.get("is_initialized") else "degraded"
        except Exception:
            status["knowledge_base"] = "error"
        
        try:
            # Test web search with simple query
            test_search = self.web_search.search_math_query("2+2")
            if hasattr(test_search, 'is_found'):
                status["web_search"] = "healthy"
            else:
                status["web_search"] = "degraded"
        except Exception as e:
            if "502" in str(e):
                status["web_search"] = "degraded"  # Temporary API issue
            else:
                status["web_search"] = "error"
        
        try:
            # Test solver (mock test)
            status["solver"] = "healthy"  # Assume healthy if no exception
        except Exception:
            status["solver"] = "error"
        
        try:
            # Test LLM/feedback processor
            status["llm"] = "healthy" if self.feedback_processor else "degraded"
        except Exception:
            status["llm"] = "error"
        
        # Overall status
        error_count = sum(1 for s in status.values() if s == "error")
        degraded_count = sum(1 for s in status.values() if s == "degraded")
        
        if error_count > 0:
            status["overall"] = "error"
        elif degraded_count > 0:
            status["overall"] = "degraded"
        else:
            status["overall"] = "healthy"
        
        return status
        
# Example usage and testing
if __name__ == "__main__":
    # Initialize orchestrator with Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    orchestrator = MathAgentOrchestrator(gemini_api_key=gemini_key)
    
    # Test query
    test_query = "Solve the quadratic equation x^2 + 5x + 6 = 0"
    
    # Process query
    print("Processing query:", test_query)
    result = orchestrator.process_query(test_query)
    print("Processing Result:", result)
    
    # Test feedback if needed
    if result.get('requires_feedback', False) and result.get('success', False):
        print("\nTesting feedback functionality...")
        feedback = "Please provide more detailed explanation of the factoring step"
        feedback_result = orchestrator.submit_feedback(
            result['solution_id'], 
            feedback
        )
        print("Feedback Result:", feedback_result)
    
    # Health check
    print("\nSystem Health Check:")
    health_status = orchestrator.health_check()
    print(health_status)