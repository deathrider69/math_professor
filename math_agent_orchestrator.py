import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

# CrewAI imports
import crewai
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pydantic import PrivateAttr

# DSPy for human-in-the-loop
import dspy

# Import existing components
from src.guardrails.input_guardrails import MathInputGuardrails
from src.guardrails.output_guardrails import MathOutputGuardrails
from src.knowledge_base.agentic_rag_kb import AIMOKnowledgeBaseComponent
from src.web_search.math_web_search import MathWebSearchComponent
from src.solver.deepseek_solver import MathProblemSolver

# Load environment variables early
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)
dspy.configure(lm=llm)

@dataclass
class MathSolutionResponse:
    """Standardized response format for math solutions"""
    solution: str
    answer: str
    source: str
    confidence: float
    requires_human_feedback: bool = False
    feedback_received: Optional[str] = None

class MathKnowledgeBaseTool(BaseTool):
    name: str = "math_knowledge_base"
    description: str = "Search the mathematical knowledge base for existing solutions"
    _kb_component: AIMOKnowledgeBaseComponent = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._kb_component = AIMOKnowledgeBaseComponent()
        self._kb_component.initialize_knowledge_base()

    def _run(self, query: str):
        result = self._kb_component.query(query)
        return {
            "success": result.get("is_present", False),
            "data": result,
            "source": "knowledge_base"
        }

class MathWebSearchTool(BaseTool):
    name: str = "math_web_search"
    description: str = "Search the web for mathematical solutions when not found in knowledge base"
    _search_component: MathWebSearchComponent = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._search_component = MathWebSearchComponent()

    def _run(self, query: str):
        result = self._search_component.search_math_query(query)
        return {
            "success": result.is_found,
            "data": {"solution": result.solution, "answer": result.answer} if result.is_found else None,
            "source": "web_search"
        }

class MathSolverTool(BaseTool):
    name: str = "math_solver"
    description: str = "Solve mathematical problems using AI when other sources fail"
    _solver_component: MathProblemSolver = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._solver_component = MathProblemSolver()

    def _run(self, query: str):
        result = self._solver_component.solve_math_problem(query)
        return {
            "success": result.get("calculated", False),
            "data": result,
            "source": "ai_solver"
        }

class HumanFeedbackSignature(dspy.Signature):
    original_solution = dspy.InputField(desc="The original mathematical solution")
    human_feedback = dspy.InputField(desc="Human feedback on the solution")
    improved_solution = dspy.OutputField(desc="Improved solution based on feedback")

class MathAgentOrchestrator:
    def __init__(self):
        self.setup_logging()
        self.llm = llm
        self.setup_guardrails()
        self.setup_tools()
        self.setup_agents()
        self.feedback_module = dspy.ChainOfThought(HumanFeedbackSignature)
        self.feedback_history = []

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_guardrails(self):
        self.input_guardrails = MathInputGuardrails()
        self.output_guardrails = MathOutputGuardrails()

    def setup_tools(self):
        self.knowledge_base_tool = MathKnowledgeBaseTool()
        self.web_search_tool = MathWebSearchTool()
        self.solver_tool = MathSolverTool()

    def setup_agents(self):
        self.kb_agent = Agent(
            role="Mathematical Knowledge Base Specialist",
            goal="Search and retrieve solutions from the mathematical knowledge base",
            backstory="You are an expert at finding relevant mathematical solutions from existing knowledge bases.",
            tools=[self.knowledge_base_tool],
            llm=self.llm,
            verbose=True
        )

        self.search_agent = Agent(
            role="Mathematical Research Specialist",
            goal="Find mathematical solutions through web search",
            backstory="You are skilled at finding accurate mathematical solutions online.",
            tools=[self.web_search_tool],
            llm=self.llm,
            verbose=True
        )

        self.solver_agent = Agent(
            role="Mathematical Problem Solver",
            goal="Solve mathematical problems step-by-step",
            backstory="You are a mathematical genius capable of solving complex problems.",
            tools=[self.solver_tool],
            llm=self.llm,
            verbose=True
        )

        self.coordinator_agent = Agent(
            role="Mathematical Solution Coordinator",
            goal="Coordinate the solution process and ensure quality responses",
            backstory="You manage the mathematical solution pipeline.",
            llm=self.llm,
            verbose=True
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        self.logger.info(f"Processing query: {query}")

        guardrail_result = self.input_guardrails.process_query(query)
        if not guardrail_result['should_process']:
            return {"success": False, "error": guardrail_result['error_message'], "stage": "input_guardrails"}

        sanitized_query = guardrail_result['sanitized_query']
        tasks = self.create_solution_tasks(sanitized_query)

        crew = Crew(
            agents=[self.kb_agent, self.search_agent, self.solver_agent, self.coordinator_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        try:
            result = crew.kickoff()
            processed_result = self.process_crew_result(result)

            if processed_result['success']:
                cleaned_solution = self.output_guardrails.clean_and_format(processed_result['solution'])
                processed_result['solution'] = cleaned_solution

            return processed_result

        except Exception as e:
            self.logger.error(f"Crew execution failed: {str(e)}")
            return {"success": False, "error": f"Solution generation failed: {str(e)}", "stage": "crew_execution"}

    def create_solution_tasks(self, query: str) -> list:
        return [
            Task(description=f"Search the mathematical knowledge base for solutions to: {query}", expected_output="Dictionary with solution status", agent=self.kb_agent),
            Task(description=f"If knowledge base search fails, search the web for solutions to: {query}", expected_output="Web search results if needed", agent=self.search_agent),
            Task(description=f"If both fail, solve the problem: {query}", expected_output="AI-generated step-by-step solution", agent=self.solver_agent),
            Task(description="Coordinate the solution process and provide the best response", expected_output="Final solution with source info", agent=self.coordinator_agent)
        ]

    def process_crew_result(self, crew_result) -> Dict[str, Any]:
        try:
            solution_data = self.parse_solution_string(crew_result) if isinstance(crew_result, str) else crew_result
            return {
                "success": True,
                "solution": solution_data.get("solution", ""),
                "answer": solution_data.get("answer", ""),
                "source": solution_data.get("source", "unknown"),
                "confidence": solution_data.get("confidence", 0.5),
                "requires_human_feedback": solution_data.get("confidence", 0.5) < 0.8
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to process crew result: {str(e)}"}

    def parse_solution_string(self, solution_str: str) -> Dict[str, Any]:
        return {
            "solution": solution_str,
            "answer": "See solution",
            "source": "crew_ai",
            "confidence": 0.7
        }

    def incorporate_human_feedback(self, solution: str, feedback: str) -> str:
        try:
            improved_solution = self.feedback_module(original_solution=solution, human_feedback=feedback)
            self.feedback_history.append({"original": solution, "feedback": feedback, "improved": improved_solution.improved_solution})
            return improved_solution.improved_solution
        except Exception as e:
            self.logger.error(f"Failed to incorporate feedback: {str(e)}")
            return solution

    def get_feedback_history(self) -> list:
        return self.feedback_history

    def health_check(self) -> Dict[str, Any]:
        status = {"guardrails": "healthy", "knowledge_base": "healthy", "web_search": "healthy", "solver": "healthy", "llm": "healthy", "overall": "healthy"}
        try:
            test_query = "What is 2+2?"
            if not self.input_guardrails.process_query(test_query)['should_process']:
                status["guardrails"] = "error"
            if not self.knowledge_base_tool._run(test_query).get("success"):
                status["knowledge_base"] = "error"
            if not self.web_search_tool._run(test_query).get("success"):
                status["web_search"] = "error"
            if not self.solver_tool._run(test_query).get("success"):
                status["solver"] = "error"
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            status["overall"] = "error"
        if any(v == "error" for v in status.values() if v != "overall"):
            status["overall"] = "degraded"
        return status
