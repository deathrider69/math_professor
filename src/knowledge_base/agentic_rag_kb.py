"""
initialize_knowdge_base()

query():
input: str - query string
output: dict - contains
    - is_present: bool - whether a similar problem was found
    - solution: str - step-by-step solution or error message
    - answer: str - final answer to the problem
    
get_status():
output: dict - contains
    - is_initialized: bool - whether the knowledge base is initialized
    - total_problems: int - number of problems in the knowledge base
    - similarity_threshold: float - current similarity threshold
    - collection_name: str - name of the ChromaDB collection
    - embedding_model: str - name of the embedding model used
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import re
from typing import Dict, Any, Tuple, Optional
import json
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMOKnowledgeBaseComponent:
    """
    A modular knowledge base component for mathematical problems using the AIMO dataset.
    Designed to be called externally by a main orchestrator.
    
    Returns structured responses with is_present, solution, and answer fields.
    """
    
    def __init__(self, 
                 collection_name: str = "aimo_math_problems",
                 db_path: str = "./aimo_vectordb",
                 similarity_threshold: float = 0.7,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the knowledge base component.
        
        Args:
            collection_name: Name of the ChromaDB collection
            db_path: Path to store the vector database
            similarity_threshold: Minimum similarity score to consider a match
            embedding_model: Sentence transformer model name
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Try to get existing collection or create new one
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "AIMO mathematical problems and solutions"}
            )
            
            # Check if collection has data
            if self.collection.count() > 0:
                self.is_initialized = True
                logger.info(f"Knowledge base loaded with {self.collection.count()} problems")
            else:
                logger.warning("Knowledge base is empty. Call initialize_knowledge_base() first.")
                
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    def initialize_knowledge_base(self, dataset_path: Optional[str] = None) -> bool:
        """
        Initialize the knowledge base with AIMO dataset.
        This should be called once to populate the database.
        
        Args:
            dataset_path: Optional local path to dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load dataset
            logger.info("Loading AIMO dataset...")
            if dataset_path and os.path.exists(dataset_path):
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_parquet("hf://datasets/AI-MO/aimo-validation-aime/data/train-00000-of-00001.parquet")
            
            logger.info(f"Loaded {len(df)} problems from dataset")
            
            # Build knowledge base
            self._build_vector_database(df)
            self.is_initialized = True
            
            logger.info("Knowledge base initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Main query method to be called by external orchestrator.
        
        Args:
            question: User's mathematical question
            
        Returns:
            Dict with keys: is_present (bool), solution (str), answer (str)
        """
        if not self.is_initialized:
            return {
                "is_present": False,
                "solution": "Knowledge base not initialized. Please call initialize_knowledge_base() first.",
                "answer": ""
            }
        
        try:
            # Search for similar problems
            similar_problems = self._search_similar_problems(question, n_results=3)
            
            if not similar_problems:
                return {
                    "is_present": False,
                    "solution": "No similar problems found in knowledge base.",
                    "answer": ""
                }
            
            # Check if best match meets threshold
            best_match = similar_problems[0]
            
            if best_match['similarity_score'] < self.similarity_threshold:
                return {
                    "is_present": False,
                    "solution": f"No sufficiently similar problems found. Best match similarity: {best_match['similarity_score']:.2f}",
                    "answer": ""
                }
            
            # Generate solution based on best match
            solution = self._generate_solution(question, best_match)
            
            return {
                "is_present": True,
                "solution": solution,
                "answer": best_match['answer']
            }
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return {
                "is_present": False,
                "solution": f"Error processing query: {str(e)}",
                "answer": ""
            }
    
    def _build_vector_database(self, df: pd.DataFrame) -> None:
        """
        Build the vector database from the AIMO dataset.
        
        Args:
            df: DataFrame containing AIMO problems
        """
        logger.info("Building vector database...")
        
        # Prepare data for vectorization
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            try:
                # Clean and prepare problem text
                problem = self._clean_text(str(row['problem']))
                solution = self._clean_text(str(row['solution']))
                
                # Create document for embedding (problem + solution context)
                document_text = f"Problem: {problem}\n\nSolution Context: {solution[:300]}..."
                
                # Create metadata
                metadata = {
                    "original_id": str(row['id']),
                    "problem": problem,
                    "solution": solution,
                    "answer": str(row['answer']),
                    "url": str(row['url']) if pd.notna(row['url']) else "",
                    "problem_type": self._classify_problem_type(problem),
                    "difficulty": self._estimate_difficulty(problem, solution)
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(f"aimo_{row['id']}")
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Create embeddings in batches
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            all_embeddings.extend(embeddings.tolist())
        
        # Add to ChromaDB
        logger.info("Adding documents to vector database...")
        self.collection.add(
            documents=documents,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully built vector database with {len(documents)} problems")
    
    def _search_similar_problems(self, query: str, n_results: int = 3) -> list:
        """
        Search for similar problems in the knowledge base.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar problems with similarity scores
        """
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # Format results with similarity scores
            similar_problems = []
            for i in range(len(results['documents'][0])):
                similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                problem_data = {
                    "id": results['ids'][0][i],
                    "problem": results['metadatas'][0][i]['problem'],
                    "solution": results['metadatas'][0][i]['solution'],
                    "answer": results['metadatas'][0][i]['answer'],
                    "url": results['metadatas'][0][i]['url'],
                    "problem_type": results['metadatas'][0][i]['problem_type'],
                    "difficulty": results['metadatas'][0][i]['difficulty'],
                    "similarity_score": similarity_score
                }
                similar_problems.append(problem_data)
            
            return similar_problems
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _generate_solution(self, query: str, best_match: Dict[str, Any]) -> str:
        """
        Generate a step-by-step solution based on the best matching problem.
        
        Args:
            query: Original user query
            best_match: Best matching problem from knowledge base
            
        Returns:
            Formatted step-by-step solution
        """
        try:
            # Extract solution steps
            solution_steps = self._extract_solution_steps(best_match['solution'])
            
            # Format the solution
            formatted_solution = f"""**Problem Type:** {best_match['problem_type'].title()}
**Difficulty Level:** {best_match['difficulty'].title()}
**Similarity Score:** {best_match['similarity_score']:.2f}

**Reference Problem:**
{best_match['problem'][:200]}...

**Step-by-Step Solution:**

{solution_steps}

**Final Answer:** {best_match['answer']}

**Source:** {best_match['url']}

---
*Note: This solution is based on a similar problem from the AIMO dataset. Please verify the approach and adapt it to your specific problem.*"""
            
            return formatted_solution
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return f"Error generating solution: {str(e)}"
    
    def _extract_solution_steps(self, solution: str) -> str:
        """
        Extract and format solution steps from the raw solution text.
        
        Args:
            solution: Raw solution text
            
        Returns:
            Formatted solution steps
        """
        # Remove excessive whitespace and clean up
        solution = re.sub(r'\s+', ' ', solution.strip())
        
        # Split into logical steps
        steps = []
        
        # Look for natural break points and step indicators
        step_indicators = [
            r'Step \d+[:\.]',
            r'\d+\.',
            r'First[,\s]',
            r'Next[,\s]',
            r'Then[,\s]',
            r'Finally[,\s]',
            r'Therefore[,\s]',
            r'Since[,\s]',
            r'Note that[,\s]',
            r'We have[,\s]',
            r'Let[,\s]'
        ]
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', solution)
        current_step = ""
        step_number = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # Skip very short sentences
                continue
            
            # Check if this sentence starts a new step
            is_new_step = False
            for pattern in step_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_new_step = True
                    break
            
            if is_new_step and current_step:
                # Save current step and start new one
                steps.append(f"**Step {step_number}:** {current_step.strip()}")
                step_number += 1
                current_step = sentence
            else:
                # Add to current step
                if current_step:
                    current_step += " " + sentence
                else:
                    current_step = sentence
        
        # Add final step
        if current_step:
            steps.append(f"**Step {step_number}:** {current_step.strip()}")
        
        # If no clear steps found, create numbered paragraphs
        if not steps:
            paragraphs = [p.strip() for p in solution.split('\n\n') if len(p.strip()) > 50]
            steps = [f"**Step {i+1}:** {para}" for i, para in enumerate(paragraphs[:5])]
        
        return '\n\n'.join(steps)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Clean up LaTeX notation for better readability
        text = text.replace('$', ' ')
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        
        return text
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify mathematical problem type."""
        problem_lower = problem.lower()
        
        type_keywords = {
            "algebra": ["polynomial", "quadratic", "equation", "function", "variable", "coefficient"],
            "geometry": ["triangle", "circle", "angle", "area", "perimeter", "polygon", "rectangle"],
            "number_theory": ["prime", "integer", "divisible", "remainder", "modular", "gcd", "lcm"],
            "combinatorics": ["combination", "permutation", "choose", "arrange", "ways", "count"],
            "probability": ["probability", "random", "chance", "expected", "outcome", "dice"],
            "calculus": ["derivative", "integral", "limit", "rate", "maximum", "minimum"]
        }
        
        scores = {}
        for ptype, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                scores[ptype] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    def _estimate_difficulty(self, problem: str, solution: str) -> str:
        """Estimate problem difficulty."""
        score = 0
        
        # Length indicators
        score += len(problem) // 150
        score += len(solution) // 800
        
        # Complexity keywords
        complex_terms = ["polynomial", "quadratic", "trigonometric", "logarithm", 
                        "system", "optimization", "proof", "theorem"]
        
        for term in complex_terms:
            if term in problem.lower():
                score += 2
            if term in solution.lower():
                score += 1
        
        if score <= 2:
            return "easy"
        elif score <= 5:
            return "medium"
        else:
            return "hard"
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status information."""
        return {
            "is_initialized": self.is_initialized,
            "total_problems": self.collection.count() if self.is_initialized else 0,
            "similarity_threshold": self.similarity_threshold,
            "collection_name": self.collection_name,
            "embedding_model": "all-MiniLM-L6-v2"
        }
