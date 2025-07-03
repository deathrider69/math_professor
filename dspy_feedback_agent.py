import dspy
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FeedbackData:
    """Data structure for feedback information"""
    original_query: str
    original_solution: str
    human_feedback: str
    refined_solution: str
    feedback_score: float
    timestamp: str

class MathSolutionSignature(dspy.Signature):
    """Signature for mathematical solution generation"""
    problem = dspy.InputField(desc="The mathematical problem to solve")
    context = dspy.InputField(desc="Additional context or constraints")
    solution = dspy.OutputField(desc="Step-by-step mathematical solution")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")

class SolutionRefinementSignature(dspy.Signature):
    """Signature for refining solutions based on feedback"""
    original_problem = dspy.InputField(desc="The original mathematical problem")
    original_solution = dspy.InputField(desc="The original solution provided")
    human_feedback = dspy.InputField(desc="Human feedback on the solution")
    refined_solution = dspy.OutputField(desc="Refined solution addressing the feedback")
    improvement_explanation = dspy.OutputField(desc="Explanation of improvements made")

class FeedbackEvaluationSignature(dspy.Signature):
    """Signature for evaluating feedback quality"""
    solution = dspy.InputField(desc="The mathematical solution")
    feedback = dspy.InputField(desc="Human feedback on the solution")
    evaluation_score = dspy.OutputField(desc="Quality score of the feedback (0-1)")
    evaluation_reasoning = dspy.OutputField(desc="Reasoning for the evaluation")

class DSPyMathFeedbackAgent:
    """
    DSPy-based agent for handling human feedback and solution refinement
    """
    
    def __init__(self, gemini_api_key: str):
        # Configure DSPy with Gemini
        self.lm = dspy.Google(model="gemini-1.5-flash", api_key=gemini_api_key)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize DSPy modules
        self.solution_generator = dspy.ChainOfThought(MathSolutionSignature)
        self.solution_refiner = dspy.ChainOfThought(SolutionRefinementSignature)
        self.feedback_evaluator = dspy.ChainOfThought(FeedbackEvaluationSignature)
        
        # Feedback storage
        self.feedback_history: List[FeedbackData] = []
        
        logger.info("DSPy Math Feedback Agent initialized")
    
    def generate_solution(self, problem: str, context: str = "") -> Dict[str, Any]:
        """Generate a mathematical solution using DSPy"""
        try:
            result = self.solution_generator(problem=problem, context=context)
            
            return {
                "solution": result.solution,
                "confidence": float(result.confidence) if result.confidence.replace('.', '').isdigit() else 0.7,
                "generated_by": "dspy_chain_of_thought"
            }
        except Exception as e:
            logger.error(f"Error in solution generation: {str(e)}")
            return {
                "solution": "Unable to generate solution",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def refine_solution_with_feedback(self, original_problem: str, original_solution: str, 
                                    human_feedback: str) -> Dict[str, Any]:
        """Refine solution based on human feedback using DSPy"""
        try:
            result = self.solution_refiner(
                original_problem=original_problem,
                original_solution=original_solution,
                human_feedback=human_feedback
            )
            
            # Store feedback data
            feedback_data = FeedbackData(
                original_query=original_problem,
                original_solution=original_solution,
                human_feedback=human_feedback,
                refined_solution=result.refined_solution,
                feedback_score=self._calculate_feedback_score(human_feedback),
                timestamp=datetime.now().isoformat()
            )
            
            self.feedback_history.append(feedback_data)
            
            return {
                "refined_solution": result.refined_solution,
                "improvement_explanation": result.improvement_explanation,
                "feedback_incorporated": True,
                "feedback_id": len(self.feedback_history) - 1
            }
            
        except Exception as e:
            logger.error(f"Error in solution refinement: {str(e)}")
            return {
                "refined_solution": original_solution,
                "improvement_explanation": "Unable to refine solution",
                "feedback_incorporated": False,
                "error": str(e)
            }
    
    def evaluate_feedback_quality(self, solution: str, feedback: str) -> Dict[str, Any]:
        """Evaluate the quality of human feedback"""
        try:
            result = self.feedback_evaluator(solution=solution, feedback=feedback)
            
            return {
                "evaluation_score": float(result.evaluation_score) if result.evaluation_score.replace('.', '').isdigit() else 0.5,
                "evaluation_reasoning": result.evaluation_reasoning,
                "feedback_quality": self._categorize_feedback_quality(
                    float(result.evaluation_score) if result.evaluation_score.replace('.', '').isdigit() else 0.5
                )
            }
            
        except Exception as e:
            logger.error(f"Error in feedback evaluation: {str(e)}")
            return {
                "evaluation_score": 0.0,
                "evaluation_reasoning": "Unable to evaluate feedback",
                "error": str(e)
            }
    
    def _calculate_feedback_score(self, feedback: str) -> float:
        """Calculate a score for feedback quality"""
        # Simple heuristic - can be enhanced with more sophisticated scoring
        score = 0.5  # Base score
        
        if len(feedback) > 10:
            score += 0.1
        
        # Check for constructive keywords
        constructive_keywords = ['explain', 'clarify', 'detail', 'step', 'show', 'help', 'understand']
        for keyword in constructive_keywords:
            if keyword.lower() in feedback.lower():
                score += 0.1
                break
        
        # Check for mathematical terms
        math_keywords = ['equation', 'formula', 'calculate', 'solve', 'proof', 'theorem']
        for keyword in math_keywords:
            if keyword.lower() in feedback.lower():
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _categorize_feedback_quality(self, score: float) -> str:
        """Categorize feedback quality based on score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Average"
        else:
            return "Poor"
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get analytics about feedback received"""
        if not self.feedback_history:
            return {"total_feedback": 0, "message": "No feedback data available"}
        
        total_feedback = len(self.feedback_history)
        avg_score = sum(f.feedback_score for f in self.feedback_history) / total_feedback
        
        quality_distribution = {
            "excellent": sum(1 for f in self.feedback_history if f.feedback_score >= 0.8),
            "good": sum(1 for f in self.feedback_history if 0.6 <= f.feedback_score < 0.8),
            "average": sum(1 for f in self.feedback_history if 0.4 <= f.feedback_score < 0.6),
            "poor": sum(1 for f in self.feedback_history if f.feedback_score < 0.4)
        }
        
        return {
            "total_feedback": total_feedback,
            "average_feedback_score": avg_score,
            "quality_distribution": quality_distribution,
            "recent_feedback": [
                {
                    "timestamp": f.timestamp,
                    "score": f.feedback_score,
                    "feedback": f.human_feedback[:100] + "..." if len(f.human_feedback) > 100 else f.human_feedback
                }
                for f in self.feedback_history[-5:]  # Last 5 feedback items
            ]
        }
    
    def train_from_feedback(self):
        """Train the agent from collected feedback (placeholder for future implementation)"""
        # This would implement training/fine-tuning based on feedback
        # For now, we'll just log the analytics
        analytics = self.get_feedback_analytics()
        logger.info(f"Feedback analytics: {analytics}")
        
        # Future: Implement actual training/optimization based on feedback patterns
        return analytics

class HumanInTheLoopManager:
    """
    Manager for human-in-the-loop operations
    """
    
    def __init__(self, feedback_agent: DSPyMathFeedbackAgent):
        self.feedback_agent = feedback_agent
        self.pending_feedback = {}  # Store solutions waiting for feedback
        
    def submit_for_feedback(self, solution_id: str, problem: str, solution: str, 
                          confidence: float) -> Dict[str, Any]:
        """Submit a solution for human feedback"""
        self.pending_feedback[solution_id] = {
            "problem": problem,
            "solution": solution,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "feedback_received": False
        }
        
        return {
            "solution_id": solution_id,
            "status": "pending_feedback",
            "requires_feedback": confidence < 0.8,  # Threshold for requiring feedback
            "message": "Solution submitted for human review"
        }
    
    def process_feedback(self, solution_id: str, feedback: str) -> Dict[str, Any]:
        """Process received human feedback"""
        if solution_id not in self.pending_feedback:
            return {"error": "Solution ID not found"}
        
        pending_item = self.pending_feedback[solution_id]
        
        # Evaluate feedback quality
        feedback_eval = self.feedback_agent.evaluate_feedback_quality(
            pending_item["solution"], feedback
        )
        
        # Refine solution based on feedback
        refinement_result = self.feedback_agent.refine_solution_with_feedback(
            pending_item["problem"],
            pending_item["solution"],
            feedback
        )
        
        # Update pending item
        pending_item["feedback_received"] = True
        pending_item["feedback"] = feedback
        pending_item["feedback_evaluation"] = feedback_eval
        pending_item["refinement_result"] = refinement_result
        
        return {
            "solution_id": solution_id,
            "feedback_processed": True,
            "feedback_quality": feedback_eval["feedback_quality"],
            "refined_solution": refinement_result["refined_solution"],
            "improvement_explanation": refinement_result["improvement_explanation"]
        }
    
    def get_pending_feedback_items(self) -> List[Dict[str, Any]]:
        """Get all items pending feedback"""
        return [
            {
                "solution_id": sid,
                "problem": item["problem"],
                "solution": item["solution"],
                "confidence": item["confidence"],
                "timestamp": item["timestamp"]
            }
            for sid, item in self.pending_feedback.items()
            if not item["feedback_received"]
        ]

# Example usage
if __name__ == "__main__":
    # Initialize the feedback agent
    feedback_agent = DSPyMathFeedbackAgent(gemini_api_key="your_gemini_api_key")
    
    # Initialize human-in-the-loop manager
    hitl_manager = HumanInTheLoopManager(feedback_agent)
    
    # Example workflow
    problem = "Solve the quadratic equation x² - 5x + 6 = 0"
    
    # Generate initial solution
    solution_result = feedback_agent.generate_solution(problem)
    print("Initial Solution:")
    print(json.dumps(solution_result, indent=2))
    
    # Submit for feedback (if confidence is low)
    if solution_result["confidence"] < 0.8:
        feedback_submission = hitl_manager.submit_for_feedback(
            "solution_001", problem, solution_result["solution"], solution_result["confidence"]
        )
        print("\nFeedback Submission:")
        print(json.dumps(feedback_submission, indent=2))
        
        # Simulate human feedback
        human_feedback = "Please show the factoring steps more clearly and verify the roots"
        
        # Process feedback
        feedback_result = hitl_manager.process_feedback("solution_001", human_feedback)
        print("\nFeedback Processing Result:")
        print(json.dumps(feedback_result, indent=2))
    
    # Get feedback analytics
    analytics = feedback_agent.get_feedback_analytics()
    print("\nFeedback Analytics:")
    print(json.dumps(analytics, indent=2))