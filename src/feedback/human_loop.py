import dspy
from typing import Dict, Any, List, Optional
import json
from datetime import datetime


class MathSolutionSignature(dspy.Signature):
    """Signature for math solution refinement"""
    original_query = dspy.InputField(desc="The original mathematical query")
    initial_solution = dspy.InputField(desc="The initial solution provided")
    human_feedback = dspy.InputField(desc="Human feedback on the solution")
    refined_solution = dspy.OutputField(desc="Refined solution based on feedback")


class MathFeedbackCollector(dspy.Module):
    """Module to collect and process human feedback"""
    
    def __init__(self):
        super().__init__()
        self.refine_solution = dspy.ChainOfThought(MathSolutionSignature)
        self.feedback_history = []
        
    def forward(self, query: str, solution: str, feedback: str) -> str:
        """Process human feedback and refine solution"""
        
        # Store feedback for learning
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "original_solution": solution,
            "feedback": feedback,
            "session_id": getattr(self, 'session_id', 'default')
        }
        self.feedback_history.append(feedback_entry)
        
        # Refine solution based on feedback
        refined = self.refine_solution(
            original_query=query,
            initial_solution=solution,
            human_feedback=feedback
        )
        
        return refined.refined_solution
    
    def get_feedback_history(self) -> List[Dict]:
        """Get the feedback history for analysis"""
        return self.feedback_history
    
    def save_feedback_history(self, filepath: str):
        """Save feedback history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def load_feedback_history(self, filepath: str):
        """Load feedback history from file"""
        try:
            with open(filepath, 'r') as f:
                self.feedback_history = json.load(f)
        except FileNotFoundError:
            self.feedback_history = []


class HumanInTheLoopManager:
    """Manager for human-in-the-loop functionality"""
    
    def __init__(self):
        self.feedback_collector = MathFeedbackCollector()
        self.pending_feedback = {}
        
    def submit_solution_for_review(self, query: str, solution: str, 
                                 answer: str, source: str) -> str:
        """Submit a solution for human review"""
        
        solution_id = f"sol_{datetime.now().timestamp()}"
        
        self.pending_feedback[solution_id] = {
            "query": query,
            "solution": solution,
            "answer": answer,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        return solution_id
    
    def process_human_feedback(self, solution_id: str, feedback: str, 
                             rating: int) -> Dict[str, Any]:
        """Process human feedback for a solution"""
        
        if solution_id not in self.pending_feedback:
            return {"error": "Solution ID not found"}
        
        solution_data = self.pending_feedback[solution_id]
        
        if rating < 3:  # If rating is low, refine the solution
            refined_solution = self.feedback_collector.forward(
                solution_data["query"],
                solution_data["solution"],
                feedback
            )
            
            result = {
                "refined": True,
                "original_solution": solution_data["solution"],
                "refined_solution": refined_solution,
                "feedback": feedback,
                "rating": rating
            }
        else:
            result = {
                "refined": False,
                "solution": solution_data["solution"],
                "feedback": feedback,
                "rating": rating
            }
        
        # Update status
        self.pending_feedback[solution_id]["status"] = "completed"
        self.pending_feedback[solution_id]["feedback"] = feedback
        self.pending_feedback[solution_id]["rating"] = rating
        
        return result
    
    def get_pending_solutions(self) -> Dict[str, Dict]:
        """Get all pending solutions for review"""
        return {k: v for k, v in self.pending_feedback.items() 
                if v["status"] == "pending"}