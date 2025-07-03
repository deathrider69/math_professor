from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from src.guardrails.input_guardrails import MathInputGuardrails
from src.guardrails.output_guardrails import MathOutputGuardrails
from src.knowledge_base.agentic_rag_kb import AIMOKnowledgeBaseComponent
from src.web_search.math_web_search import MathSearchResult
from src.solver.deepseek_solver import MathProblemSolver


class MathProcessingTool(BaseTool):
    """Tool for processing mathematical queries through the pipeline"""
    
    name: str = "math_processor"
    description: str = "Process mathematical queries through guardrails, knowledge base, web search, and solver"
    
    def __init__(self):
        super().__init__()
        self.input_guardrails = MathInputGuardrails()
        self.output_guardrails = MathOutputGuardrails()
        self.knowledge_base = AIMOKnowledgeBaseComponent()
        self.web_search = MathSearchResult()
        self.solver = MathProblemSolver()
        
    def _run(self, query: str) -> Dict[str, Any]:
        """Process the mathematical query through the pipeline"""
        result = {
            "original_query": query,
            "solution": "",
            "answer": "",
            "source": "",
            "processed": False,
            "error": None
        }
        
        try:
            # Step 1: Input Guardrails
            guardrail_result = self.input_guardrails.process_query(query)
            
            if not guardrail_result["should_process"]:
                result["error"] = guardrail_result["error_message"]
                return result
            
            sanitized_query = guardrail_result["sanitized_query"]
            
            # Step 2: Knowledge Base Query
            kb_result = self.knowledge_base.query(sanitized_query)
            
            if kb_result["is_present"]:
                result["solution"] = kb_result["solution"]
                result["answer"] = kb_result["answer"]
                result["source"] = "knowledge_base"
                result["processed"] = True
                return result
            
            # Step 3: Web Search
            search_result = self.web_search.search_math_query(sanitized_query)
            
            if search_result["is_found"]:
                result["solution"] = search_result["solution"]
                result["answer"] = str(search_result["answer"])
                result["source"] = "web_search"
                result["processed"] = True
                return result
            
            # Step 4: DeepSeek Solver
            solver_result = self.solver.solve_math_problem(sanitized_query)
            
            if solver_result["calculated"]:
                result["solution"] = solver_result["solution"]
                result["answer"] = solver_result["answer"]
                result["source"] = "deepseek_solver"
                result["processed"] = True
                return result
            
            result["error"] = "Unable to solve the problem using any available method"
            return result
            
        except Exception as e:
            result["error"] = f"Error processing query: {str(e)}"
            return result


class MathProfessorCrew:
    """Main CrewAI orchestration class for the Math Professor system"""
    
    def __init__(self):
        self.math_tool = MathProcessingTool()
        self.output_guardrails = MathOutputGuardrails()
        self.setup_agents()
        
    def setup_agents(self):
        """Setup the CrewAI agents"""
        
        # Input Processing Agent
        self.input_agent = Agent(
            role='Math Input Processor',
            goal='Process and validate mathematical queries through guardrails',
            backstory='You are an expert at understanding and validating mathematical queries. '
                     'Your job is to ensure queries are safe and properly formatted.',
            verbose=True,
            allow_delegation=False,
            tools=[self.math_tool]
        )
        
        # Knowledge Retrieval Agent
        self.knowledge_agent = Agent(
            role='Knowledge Base Specialist',
            goal='Retrieve relevant mathematical solutions from the knowledge base',
            backstory='You are an expert at searching and retrieving mathematical knowledge. '
                     'You know how to find similar problems and their solutions.',
            verbose=True,
            allow_delegation=False
        )
        
        # Web Search Agent
        self.search_agent = Agent(
            role='Mathematical Web Researcher',
            goal='Search the web for mathematical solutions when knowledge base fails',
            backstory='You are an expert at finding mathematical solutions online. '
                     'You know how to search for and validate mathematical content.',
            verbose=True,
            allow_delegation=False
        )
        
        # Solver Agent
        self.solver_agent = Agent(
            role='Mathematical Problem Solver',
            goal='Solve mathematical problems using advanced AI techniques',
            backstory='You are an expert mathematical problem solver. '
                     'You can solve complex mathematical problems step-by-step.',
            verbose=True,
            allow_delegation=False
        )
        
        # Output Formatting Agent
        self.output_agent = Agent(
            role='Math Output Formatter',
            goal='Format and clean mathematical solutions for student presentation',
            backstory='You are an expert at presenting mathematical solutions clearly. '
                     'You know how to format solutions for easy student understanding.',
            verbose=True,
            allow_delegation=False
        )
        
    def create_tasks(self, query: str):
        """Create tasks for the crew"""
        
        # Main processing task
        process_task = Task(
            description=f"""
            Process the mathematical query: "{query}"
            
            Follow these steps:
            1. Validate the query through input guardrails
            2. Search the knowledge base for existing solutions
            3. If not found, search the web for solutions
            4. If still not found, use the AI solver
            5. Format the final output appropriately
            
            Provide a comprehensive step-by-step solution that a student can understand.
            """,
            agent=self.input_agent,
            tools=[self.math_tool],
            expected_output="A detailed mathematical solution with step-by-step explanation"
        )
        
        return [process_task]
    
    def solve_math_problem(self, query: str) -> Dict[str, Any]:
        """Main method to solve mathematical problems"""
        
        # Create tasks
        tasks = self.create_tasks(query)
        
        # Create crew
        crew = Crew(
            agents=[self.input_agent, self.knowledge_agent, self.search_agent, 
                   self.solver_agent, self.output_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Execute the crew
        raw_result = crew.kickoff()
        
        # Process through the math tool directly for structured output
        structured_result = self.math_tool._run(query)
        
        if structured_result["processed"]:
            # Clean and format the solution
            cleaned_solution = self.output_guardrails.clean_and_format(
                structured_result["solution"]
            )
            
            return {
                "success": True,
                "solution": cleaned_solution,
                "answer": structured_result["answer"],
                "source": structured_result["source"],
                "raw_crew_output": str(raw_result),
                "error": None
            }
        else:
            return {
                "success": False,
                "solution": "",
                "answer": "",
                "source": "",
                "raw_crew_output": str(raw_result),
                "error": structured_result.get("error", "Unknown error occurred")
            }