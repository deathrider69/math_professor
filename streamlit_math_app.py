import streamlit as st
import sys
import os
from typing import Dict, Any
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add the orchestrator to the path
sys.path.append(os.path.dirname(__file__))

from math_agent_orchestrator import MathAgentOrchestrator, SolutionSource

# Configure Streamlit page
st.set_page_config(
    page_title="Mathematical Professor AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .solution-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .feedback-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .source-kb {
        background-color: #d4edda;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    
    .source-web {
        background-color: #cce7ff;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    
    .source-solver {
        background-color: #ffe6cc;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    
    .source-feedback {
        background-color: #f8d7da;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MathAgentOrchestrator()

if 'solution_history' not in st.session_state:
    st.session_state.solution_history = []

if 'current_solution_id' not in st.session_state:
    st.session_state.current_solution_id = None

def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_source_class(source: str) -> str:
    """Get CSS class based on solution source"""
    source_map = {
        "knowledge_base": "source-kb",
        "web_search": "source-web",
        "solver": "source-solver",
        "human_feedback": "source-feedback"
    }
    return source_map.get(source, "source-solver")

def format_solution_display(result: Dict[str, Any]) -> None:
    """Format and display solution result"""
    if not result.get('success', False):
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    confidence = result.get('confidence', 0.0)
    source = result.get('source', 'unknown')
    
    # Solution header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Solution ID:** `{result.get('solution_id', 'N/A')}`")
    
    with col2:
        confidence_class = get_confidence_class(confidence)
        st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence:.1%}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        source_class = get_source_class(source)
        st.markdown(f"**Source:** <span class='{source_class}'>{source.replace('_', ' ').title()}</span>", 
                   unsafe_allow_html=True)
    
    # Solution content
    st.markdown(f"""
    <div class="solution-box">
        <h4>Step-by-Step Solution:</h4>
        {result.get('solution', 'No solution available')}
    </div>
    """, unsafe_allow_html=True)
    
    # Final answer
    if result.get('answer'):
        st.markdown(f"**🎯 Final Answer:** `{result.get('answer')}`")
    
    # Reasoning
    if result.get('reasoning'):
        st.markdown(f"**💡 Reasoning:** {result.get('reasoning')}")
    
    # Feedback indicator
    if result.get('requires_feedback', False):
        st.markdown("""
        <div class="feedback-box">
            <h4>🔄 Human Feedback Recommended</h4>
            <p>This solution would benefit from human validation. Please provide feedback below.</p>
        </div>
        """, unsafe_allow_html=True)

def display_analytics_dashboard():
    """Display analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    if not st.session_state.solution_history:
        st.info("No solutions processed yet. Solve some math problems to see analytics!")
        return
    
    # Create dataframe from history
    df = pd.DataFrame(st.session_state.solution_history)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Solutions", len(df))
    
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        feedback_count = df['requires_feedback'].sum()
        st.metric("Feedback Requests", feedback_count)
    
    with col4:
        success_rate = df['success'].mean()
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Source distribution
        source_counts = df['source'].value_counts()
        fig_source = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Solution Sources"
        )
        st.plotly_chart(fig_source, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig_conf = px.histogram(
            df, 
            x='confidence', 
            nbins=10,
            title="Confidence Distribution"
        )
        st.plotly_chart(fig_conf, use_container_width=True)

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🧮 Mathematical Professor AI</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Mathematical Professor AI! This intelligent system can solve mathematical problems 
    by checking its knowledge base, performing web searches, and using advanced solvers. 
    It also incorporates human feedback for continuous improvement.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Navigation
        page = st.selectbox(
            "Navigate",
            ["Solve Problems", "Solution History", "Analytics", "System Status"]
        )
        
        st.markdown("---")
        
        # System info
        st.subheader("System Info")
        st.markdown(f"**Sessions:** {len(st.session_state.solution_history)}")
        if st.session_state.current_solution_id:
            st.markdown(f"**Current Solution:** `{st.session_state.current_solution_id}`")
        
        # Clear history
        if st.button("Clear History"):
            st.session_state.solution_history = []
            st.session_state.current_solution_id = None
            st.success("History cleared!")
    
    # Main content based on selected page
    if page == "Solve Problems":
        st.header("Problem Solving")
        
        # Input form
        with st.form("math_problem_form"):
            problem = st.text_area(
                "Enter your mathematical problem:",
                placeholder="e.g., Solve x^2 + 5x + 6 = 0",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit = st.form_submit_button("Solve Problem")
            with col2:
                if submit and not problem.strip():
                    st.error("Please enter a mathematical problem!")
        
        # Process problem
        if submit and problem.strip():
            with st.spinner("Processing your problem..."):
                result = st.session_state.orchestrator.process_query(problem)
                
                # Store in history
                history_entry = {
                    'timestamp': datetime.now(),
                    'problem': problem,
                    'success': result.get('success', False),
                    'confidence': result.get('confidence', 0.0),
                    'source': result.get('source', 'unknown'),
                    'requires_feedback': result.get('requires_feedback', False),
                    'solution_id': result.get('solution_id')
                }
                st.session_state.solution_history.append(history_entry)
                st.session_state.current_solution_id = result.get('solution_id')
                
                # Display result
                format_solution_display(result)
        
        # Feedback section
        if st.session_state.current_solution_id:
            st.markdown("---")
            st.subheader("Provide Feedback")
            
            with st.form("feedback_form"):
                feedback = st.text_area(
                    "Your feedback on the solution:",
                    placeholder="e.g., Please explain the factoring step in more detail...",
                    height=80
                )
                
                feedback_submit = st.form_submit_button("📤 Submit Feedback")
            
            if feedback_submit and feedback.strip():
                with st.spinner("🔄 Processing feedback..."):
                    feedback_result = st.session_state.orchestrator.submit_feedback(
                        st.session_state.current_solution_id,
                        feedback
                    )
                    
                    if feedback_result.get('success', False):
                        st.success("Feedback processed successfully!")
                        st.markdown(f"""
                        <div class="solution-box">
                            <h4>🔄 Improved Solution:</h4>
                            {feedback_result.get('improved_solution', 'No improvement available')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"**New Confidence:** {feedback_result.get('confidence', 0.0):.1%}")
                    else:
                        st.error(f"Error processing feedback: {feedback_result.get('error', 'Unknown error')}")
    
    elif page == "Solution History":
        st.header("📚 Solution History")
        
        if not st.session_state.solution_history:
            st.info("No solutions in history yet.")
            return
        
        # Display history
        for i, entry in enumerate(reversed(st.session_state.solution_history)):
            with st.expander(f"Solution {len(st.session_state.solution_history) - i}: {entry['problem'][:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Problem:** {entry['problem']}")
                    st.markdown(f"**Timestamp:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**Success:** {'Successful' if entry['success'] else 'Failure'}")
                
                with col2:
                    st.markdown(f"**Confidence:** {entry['confidence']:.1%}")
                    st.markdown(f"**Source:** {entry['source'].replace('_', ' ').title()}")
                    st.markdown(f"**Feedback Required:** {'Yes' if entry['requires_feedback'] else 'No'}")
                
                if entry['solution_id']:
                    if st.button(f"View Details", key=f"view_{entry['solution_id']}"):
                        details = st.session_state.orchestrator.get_solution_history(entry['solution_id'])
                        if details.get('success', False):
                            st.json(details)
    
    elif page == "Analytics":
        display_analytics_dashboard()
    
    elif page == "System Status":
        st.header("System Status")
        
        # Component status
        st.subheader("Component Health")
        
        components = [
            ("Input Guardrails", "Active"),
            ("Output Guardrails", "Active"),
            ("Knowledge Base", "Active"),
            ("Web Search", "Active"),
            ("Math Solver", "Active"),
            ("Feedback Processor", "Active")
        ]
        
        for component, status in components:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{component}**")
            with col2:
                st.markdown(status)
        
        # System metrics
        st.subheader("System Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cache Size", len(st.session_state.orchestrator.solution_cache))
        
        with col2:
            st.metric("Active Sessions", 1)
        
        with col3:
            st.metric("System Uptime", "Active")

if __name__ == "__main__":
    main()