
# Final Proposal: Math Professor Agent

## Proposal & Working Demonstration

[Drive link: Proposal and Working Demo](https://drive.google.com/drive/u/0/folders/1JFEr_-VogikcAC7vH556gtFE0dIIVYcE)

---

## Input and Output Guardrails

Guardrails have been implemented to monitor the user input and control the agentic system’s output.

### Input Guardrails

The input guardrails serve the following purposes:

- To check if the query is math-related or not.  
- To sanitize the query, i.e., replace:
  - Variable symbols with a constant symbol (e.g., `x` and `*` for multiplication, `pi` and `π`).  
  - HTML entities (e.g., `&lt;` and `<`).  
- To remove injection elements to protect the database.  
- To detect harmful content (vulgar language, profanity, etc.).

**Function: `process_query()`**

| Input | Output |
|-------|--------|
| `str` - raw user query | `dict` containing:<br> - `original_query`: `str` - the original query<br> - `sanitized_query`: `str` - sanitized version of the query<br> - `should_process`: `bool` - whether the query is valid for processing<br> - `error_message`: `str or None` - error message if validation fails |

---

### Output Guardrails

The output guardrails serve the following purposes:

- To remove harmful scripts/tags (e.g., `<embed>`, `<object>`, `<import>`, `exec`).  
- To format output in LaTeX.  

**Function: `clean_and_format()`**

| Input | Output |
|-------|--------|
| `str` - raw math output string | `str` - cleaned and formatted math output string |

**Note:**  
Prior to this, the usage of [NVIDIA's nemoguardrails](https://developer.nvidia.com/nemoguardrails) was explored for setting this up.  
- Advantageous as guardrails can be configured for LLMs (using **LLMRails**).  
- Faced integration difficulties with the current agent system.  
- Recommended as a potential future improvement.

---

## Knowledge Base

For the knowledge base, **ChromaDB** has been used as the vector store.

- The `aimo_math` dataset is used as it is a standard dataset for training mathematical models.  
- The dataset contains **step-by-step solutions** along with answers.  
- This allows for **direct retrieval** of steps and answers to queries without requiring a web search or solver-agent call.  
- Uses `similarity_scores` for retrieval.  

**Function: `query()`**

| Input | Output |
|-------|--------|
| `str` - query string | `dict` containing:<br> - `is_present`: `bool` - whether a similar problem was found<br> - `solution`: `str` - step-by-step solution or error message<br> - `answer`: `str` - final answer to the problem |

---

## Web Search Capabilities

The system uses **TavilyClient** for web search.

- The user’s query is first enhanced to improve mathematical relevance.  
- Web scraping is performed primarily from math domains:  
  - `wolfram.com`  
  - `symbolab.com`  
  - `mathway.com`  
  - `khanacademy.org`  
  - `brilliant.org`  
  - `stackexchange.com`  
  - `math.stackexchange.com`  

- Math content is extracted from scraped results.  
- A **step-by-step solution** and final **numerical result** are generated.  
- To prevent inaccurate results, a **high confidence threshold** is set for TavilyClient, due to the sensitivity of mathematical symbol mismatches.  

**Function: `search_math_query()`**

| Input | Output |
|-------|--------|
| `str` - Mathematical query string | `dict` containing:<br> - `solution`: `str` - Step-by-step solution<br> - `answer`: `float` - Numeric answer<br> - `is_found`: `bool` - Whether a solution was found |

---

## Human-in-the-Loop Architecture

Implemented using the **DSPy** library.

- When a user submits feedback via `submit_feedback()`, the `FeedbackProcessor`:  
  - Passes the original solution and human feedback to **DSPy’s Chain of Thought**.  
  - Generates an improved solution and updated confidence score.  
  - Updates the cached solution with the refined response.  
  - Marks the source as `HUMAN_FEEDBACK` for traceability.  

This enables direct **human-in-the-loop** refinement, where human validation or correction improves AI output quality.

---

## Fallback Agent

In case both Knowledge Base querying and Web Search fail or do not meet the confidence threshold, an **Internal Solver** is used.

- The internal solver uses **deepseek-r1**, deployed via **NVIDIA NIM microservices**.  
- Chosen due to its high reasoning capabilities:  
  - **91%** on AIME benchmark.  
  - **79%** on HMMT dataset.  

### Alternative Solver:

- The solver can be switched to **fathom-r1-14b**, an indigenously developed model:  
  - Based on **deepseek-r1-distilled-qwen**.  
  - Outperforms `o3-mini` and matches `o4-mini` performance.

This ensures an answer is provided even if Knowledge Base and Web Search mechanisms fail.

---

## Summary

The Math Professor Agent integrates multiple layers of:
- **Input/Output Guardrails**  
- **Vector Database Retrieval (ChromaDB)**  
- **Targeted Web Search (TavilyClient)**  
- **Human Feedback Loops (DSPy)**  
- **High-Reasoning Fallback Agent (deepseek-r1 or fathom-r1-14b)**  

Together, these components create a reliable, secure, and accurate mathematical problem-solving system.

--- 
