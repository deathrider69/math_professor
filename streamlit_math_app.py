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

#streamlit page
st.set_page_config(
    page_title="Mathematical Professor AI - LangGraph Powered",
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
        color: #000000;
    }
    
    .solution-box h4 {
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .solution-box p {
        color: #000000;
    }
    
    .feedback-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #000000;
    }
    
    .feedback-box h4 {
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .feedback-box p {
        color: #000000;
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
        color: #000000;
    }
    
    .source-web {
        background-color: #cce7ff;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #000000;
    }
    
    .source-solver {
        background-color: #ffe6cc;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #000000;
    }
    
    .source-feedback {
        background-color: #f8d7da;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def get_orchestrator():
    """Get or create orchestrator instance"""
    gemini_api_key = os.getenv("GEMINI_API_KEY") or st.session_state.get('gemini_api_key')
    if gemini_api_key:
        return MathAgentOrchestrator(gemini_api_key=gemini_api_key)
    return None

if 'solution_history' not in st.session_state:
    st.session_state.solution_history = []

if 'current_solution_id' not in st.session_state:
    st.session_state.current_solution_id = None

if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

if 'feedback_mode' not in st.session_state:
    st.session_state.feedback_mode = False

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
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        if result.get('stage'):
            st.info(f"Error occurred at stage: {result.get('stage')}")
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
    #if result.get('answer'):
    #    st.markdown(f"**Final Answer:** `{result.get('answer')}`")
    
    # Reasoning
    if result.get('reasoning'):
        st.markdown(f"**Reasoning:** {result.get('reasoning')}")
    
    # Processing step
    if result.get('current_step'):
        st.markdown(f"**Processing Step:** {result.get('current_step').replace('_', ' ').title()}")
    
    # Feedback indicator
    if result.get('requires_feedback', False):
        st.markdown("""
        <div class="feedback-box">
            <h4>Human Feedback Recommended</h4>
            <p>This solution would benefit from human validation. Please provide feedback below.</p>
        </div>
        """, unsafe_allow_html=True)

def display_analytics_dashboard():
    """Display analytics dashboard"""
    st.subheader("Analytics Dashboard")
    
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
    st.markdown('<h1 class="main-header">🧮 Math-Professor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Hello! I am your Math-Professor agent powered by **LangGraph** and **DSPy**! I can solve mathematical problems 
    through a sophisticated workflow including knowledge base search, web search, and AI solving with your feedback integration.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("System Configuration")
        
        # API Key Configuration
        st.subheader("API Configuration")
        gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get('gemini_api_key', ''),
            type="password",
            help="Required for DSPy feedback processing"
        )
        
        if gemini_key != st.session_state.get('gemini_api_key', ''):
            st.session_state.gemini_api_key = gemini_key
            # Clear cache to reinitialize orchestrator
            st.cache_resource.clear()
        
        # System Status
        orchestrator = get_orchestrator()
        if orchestrator:
            st.success("System Initialized Successfully.")
            
            # Health Check
            if st.button("Health Check"):
                with st.spinner("Checking system health..."):
                    health_status = orchestrator.health_check()
                    
                    for component, status in health_status.items():
                        if status == "healthy":
                            st.success(f"✅ {component.replace('_', ' ').title()}")
                        elif status == "degraded":
                            st.warning(f"⚠️ {component.replace('_', ' ').title()}")
                        else:
                            st.error(f"❌ {component.replace('_', ' ').title()}")
        else:
            st.error("System Not Initialized")
            st.info("Please provide Gemini API key to enable feedback processing")
        
        st.markdown("---")
        
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
        
        # System status indicators
        if orchestrator:
            health_status = orchestrator.health_check()
            web_status = health_status.get("web_search", "unknown")
            
            if web_status == "healthy":
                st.success("Web search: Available")
            elif web_status == "degraded":
                st.warning("Web search: Temporarily unavailable")
            else:
                st.error("Web search: Error")
        
        # Feedback Mode Toggle
        if st.session_state.current_solution_id and orchestrator:
            st.session_state.feedback_mode = st.checkbox("Enable Feedback Mode", st.session_state.feedback_mode)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.solution_history = []
            st.session_state.current_solution_id = None
            st.session_state.feedback_mode = False
            st.success("History cleared!")
    
    # Main content
    if not orchestrator:
        st.warning("System not fully initialized. Some features may be limited without Gemini API key.")
        st.info("You can still solve problems, but feedback processing will be unavailable.")
        
        # Create a basic orchestrator without feedback
        try:
            orchestrator = MathAgentOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize basic system: {e}")
            return
    
    # Main content based on selected page
    if page == "Solve Problems":
        st.header("Problem Solving")
        
        # Feedback Mode Interface
        if st.session_state.feedback_mode and st.session_state.current_solution_id:
            st.markdown("### Feedback Mode")
            
            with st.form("feedback_form"):
                feedback = st.text_area(
                    "Provide feedback on the current solution:",
                    placeholder="e.g., The solution is correct but could be more detailed in the factoring step...",
                    height=100
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    submit_feedback = st.form_submit_button("Submit Feedback")
                with col2:
                    cancel_feedback = st.form_submit_button("Cancel Feedback")
            
            if cancel_feedback:
                st.session_state.feedback_mode = False
                st.rerun()
            
            if submit_feedback and feedback.strip():
                with st.spinner("Processing feedback with DSPy..."):
                    feedback_result = orchestrator.submit_feedback(
                        st.session_state.current_solution_id,
                        feedback
                    )
                    
                    if feedback_result.get('success', False):
                        st.success("Feedback processed successfully!")
                        
                        # Display improved solution
                        st.markdown("### Improved Solution")
                        st.markdown(f"""
                        <div class="solution-box">
                            <h4>Refined Solution:</h4>
                            {feedback_result.get('improved_solution', 'No improvement available')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("New Confidence", f"{feedback_result.get('confidence', 0.0):.1%}")
                        with col2:
                            st.metric("Source", "Human Feedback")
                        
                        # Update history
                        for entry in st.session_state.solution_history:
                            if entry.get('solution_id') == st.session_state.current_solution_id:
                                entry['feedback_provided'] = True
                                entry['improved_solution'] = feedback_result.get('improved_solution')
                                entry['final_confidence'] = feedback_result.get('confidence', 0.0)
                                break
                        
                        st.session_state.feedback_mode = False
                        
                    else:
                        st.error(f"Error processing feedback: {feedback_result.get('error', 'Unknown error')}")
            
            st.markdown("---")
        
        # Problem Input Form
        with st.form("math_problem_form"):
            problem = st.text_area(
                "Enter your mathematical problem:",
                placeholder="e.g., Solve x^2 + 5x + 6 = 0",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit = st.form_submit_button("� Solve Problem")
            with col2:
                if submit and not problem.strip():
                    st.error("Please enter a mathematical problem!")
        
        # Process problem
        if submit and problem.strip():
            with st.spinner("🔄 Processing through LangGraph workflow..."):
                # Show workflow steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Validating input...")
                progress_bar.progress(20)
                
                status_text.text("Searching knowledge base...")
                progress_bar.progress(40)
                
                status_text.text("Performing web search...")
                progress_bar.progress(60)
                
                status_text.text("AI solving...")
                progress_bar.progress(80)
                
                status_text.text("Formatting output...")
                progress_bar.progress(100)
                
                try:
                    result = orchestrator.process_query(problem)
                except Exception as e:
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error(f"An error occurred while processing your query: {str(e)}")
                    if "502" in str(e) or "Tavily" in str(e):
                        st.info("Web search is temporarily unavailable, but the AI solver is still working. Please try again.")
                    st.stop()
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Store in history
                history_entry = {
                    'timestamp': datetime.now(),
                    'problem': problem,
                    'success': result.get('success', False),
                    'confidence': result.get('confidence', 0.0),
                    'source': result.get('source', 'unknown'),
                    'requires_feedback': result.get('requires_feedback', False),
                    'solution_id': result.get('solution_id'),
                    'current_step': result.get('current_step', ''),
                    'feedback_provided': False
                }
                st.session_state.solution_history.append(history_entry)
                st.session_state.current_solution_id = result.get('solution_id')
                
                # Display result
                format_solution_display(result)
                
                # Enable feedback mode if recommended
                if result.get('requires_feedback', False) and result.get('success', False):
                    if st.button("🔄 Provide Feedback"):
                        st.session_state.feedback_mode = True
                        st.rerun()
    
    elif page == "Solution History":
        st.header("Solution History")
        
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
                    st.markdown(f"**Success:** {'Successful' if entry['success'] else 'Failed'}")
                    st.markdown(f"**Processing Step:** {entry.get('current_step', 'Unknown').replace('_', ' ').title()}")
                
                with col2:
                    #st.markdown(f"**Confidence:** {entry['confidence']:.1%}")
                    st.markdown(f"**Source:** {entry['source'].replace('_', ' ').title()}")
                    st.markdown(f"**Feedback Required:** {'Yes' if entry['requires_feedback'] else 'No'}")
                    st.markdown(f"**Feedback Provided:** {'Yes' if entry.get('feedback_provided', False) else 'No'}")
                
                if entry.get('improved_solution'):
                    st.markdown("**Improved Solution Available** ✨")
                    if st.button(f"View Improved Solution", key=f"improved_{entry['solution_id']}"):
                        st.markdown(f"""
                        <div class="solution-box">
                            <h4>Human-Refined Solution:</h4>
                            {entry['improved_solution']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"**Final Confidence:** {entry.get('final_confidence', entry['confidence']):.1%}")
                
                if entry['solution_id'] and orchestrator:
                    if st.button(f"View Full Details", key=f"view_{entry['solution_id']}"):
                        details = orchestrator.get_solution_history(entry['solution_id'])
                        if details.get('success', False):
                            st.json(details)
    
    elif page == "Analytics":
        display_analytics_dashboard()
    
    elif page == "System Status":
        st.header("System Status")
        
        if not orchestrator:
            st.error("System not initialized")
            return
        
        # Component status
        st.subheader("Component Health")
        
        with st.spinner("Checking system health..."):
            health_status = orchestrator.health_check()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Components:**")
            core_components = ['guardrails', 'knowledge_base', 'overall']
            for component in core_components:
                status = health_status.get(component, 'unknown')
                if status == 'healthy':
                    st.success(f"✅ {component.replace('_', ' ').title()}")
                elif status == 'degraded':
                    st.warning(f"⚠️ {component.replace('_', ' ').title()}")
                else:
                    st.error(f"❌ {component.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**External Services:**")
            external_components = ['web_search', 'solver', 'llm']
            for component in external_components:
                status = health_status.get(component, 'unknown')
                if status == 'healthy':
                    st.success(f"✅ {component.replace('_', ' ').title()}")
                elif status == 'degraded':
                    st.warning(f"⚠️ {component.replace('_', ' ').title()}")
                else:
                    st.error(f"❌ {component.replace('_', ' ').title()}")
        
        # System metrics
        st.subheader("📊 System Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cache Size", len(orchestrator.solution_cache))
        
        with col2:
            feedback_history = orchestrator.get_feedback_history()
            st.metric("Feedback Sessions", len(feedback_history))
        
        with col3:
            successful_solutions = sum(1 for h in st.session_state.solution_history if h.get('success'))
            st.metric("Successful Solutions", successful_solutions)
        
        with col4:
            avg_confidence = sum(h.get('confidence', 0) for h in st.session_state.solution_history) / len(st.session_state.solution_history) if st.session_state.solution_history else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # LangGraph Workflow Visualization
        #st.subheader("LangGraph Workflow")
        #st.markdown("""
        #**Current Workflow Steps:**
        #1. **Input Validation** - Guardrails validate and sanitize input
        #2. **Knowledge Search** - Search AIMO dataset for similar problems
        #3. **Web Search** - Tavily search for online solutions (if needed)
        #4. **AI Solver** - DeepSeek solver as fallback (if needed)
        #5. **Output Formatting** - Clean and format final solution
        #6. **Feedback Processing** - DSPy-based human feedback integration
        #""")
        
        # Recent feedback history
        if feedback_history:
            st.subheader("Recent Feedback History")
            for i, feedback_item in enumerate(feedback_history[-3:]):  # Show last 3
                with st.expander(f"Feedback Session {len(feedback_history) - 2 + i}"):
                    st.markdown(f"**Timestamp:** {feedback_item.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Confidence:** {feedback_item.get('confidence', 0):.1%}")
                    st.markdown(f"**Feedback:** {feedback_item.get('feedback', 'No feedback')[:100]}...")


if __name__ == "__main__":
    main()