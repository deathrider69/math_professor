import streamlit as st
import os
import json
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import the orchestrator
from math_agent_orchestrator import MathAgentOrchestrator

# Page configuration
st.set_page_config(
    page_title="Mathematical AI Professor",
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .solution-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .feedback-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-degraded {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'feedback_mode' not in st.session_state:
    st.session_state.feedback_mode = False
if 'current_solution' not in st.session_state:
    st.session_state.current_solution = None

def initialize_orchestrator():
    """Initialize the math agent orchestrator"""
    try:
        api_key = st.session_state.get('gemini_api_key', '')
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar")
            return False
        
        with st.spinner("Initializing Mathematical AI Professor..."):
            st.session_state.orchestrator = MathAgentOrchestrator()
            st.success("Mathematical AI Professor initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {str(e)}")
        return False

def display_solution(result):
    """Display the mathematical solution"""
    st.markdown('<div class="solution-container">', unsafe_allow_html=True)
    
    if result.get('success'):
        st.markdown("### 📚 Solution")
        st.write(result.get('solution', 'No solution provided'))
        
        if result.get('answer'):
            st.markdown("### 🎯 Final Answer")
            st.code(result.get('answer'), language='text')
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source", result.get('source', 'unknown'))
        with col2:
            confidence = result.get('confidence', 0) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        with col3:
            feedback_needed = result.get('requires_human_feedback', False)
            st.metric("Feedback Needed", "Yes" if feedback_needed else "No")
        
        # Store current solution for feedback
        st.session_state.current_solution = result
        
        # Show feedback option if needed
        if result.get('requires_human_feedback', False):
            st.markdown("### 💡 This solution could benefit from human feedback")
            if st.button("Provide Feedback", key="feedback_btn"):
                st.session_state.feedback_mode = True
                st.rerun()
    else:
        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
        st.info(f"Error occurred at stage: {result.get('stage', 'unknown')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_feedback_interface():
    """Display the human feedback interface"""
    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
    st.markdown("### 🔄 Human Feedback")
    
    if st.session_state.current_solution:
        st.write("**Original Solution:**")
        st.write(st.session_state.current_solution.get('solution', ''))
        
        feedback = st.text_area(
            "Please provide your feedback to improve this solution:",
            placeholder="e.g., The solution is correct but could be more detailed in the factoring step...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Feedback", key="submit_feedback"):
                if feedback.strip():
                    with st.spinner("Incorporating feedback..."):
                        improved_solution = st.session_state.orchestrator.incorporate_human_feedback(
                            st.session_state.current_solution['solution'],
                            feedback
                        )
                        st.success("Feedback incorporated successfully!")
                        st.write("**Improved Solution:**")
                        st.write(improved_solution)
                        
                        # Update query history
                        for item in st.session_state.query_history:
                            if item['timestamp'] == st.session_state.current_solution.get('timestamp'):
                                item['feedback'] = feedback
                                item['improved_solution'] = improved_solution
                                break
                else:
                    st.warning("Please provide feedback before submitting")
        
        with col2:
            if st.button("Cancel Feedback", key="cancel_feedback"):
                st.session_state.feedback_mode = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_system_status():
    """Display system health status"""
    if st.session_state.orchestrator:
        with st.spinner("Checking system status..."):
            status = st.session_state.orchestrator.health_check()
        
        st.markdown("### 🔍 System Status")
        
        # Overall status
        overall_status = status.get('overall', 'unknown')
        if overall_status == 'healthy':
            st.success("✅ System is healthy")
        elif overall_status == 'degraded':
            st.warning("⚠️ System is degraded")
        else:
            st.error("❌ System has errors")
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Components:**")
            for component in ['guardrails', 'knowledge_base']:
                status_val = status.get(component, 'unknown')
                status_class = f"status-{status_val}"
                st.markdown(f"<span class='{status_class}'>{component.title()}: {status_val}</span>", 
                          unsafe_allow_html=True)
        
        with col2:
            st.write("**External Services:**")
            for component in ['web_search', 'solver', 'llm']:
                status_val = status.get(component, 'unknown')
                status_class = f"status-{status_val}"
                st.markdown(f"<span class='{status_class}'>{component.title()}: {status_val}</span>", 
                          unsafe_allow_html=True)

def display_analytics():
    """Display analytics and insights"""
    if st.session_state.query_history:
        st.markdown("### 📊 Analytics")
        
        # Query statistics
        total_queries = len(st.session_state.query_history)
        successful_queries = sum(1 for q in st.session_state.query_history if q.get('success'))
        feedback_provided = sum(1 for q in st.session_state.query_history if q.get('feedback'))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Feedback Provided", feedback_provided)
        
        # Source distribution
        sources = [q.get('source', 'unknown') for q in st.session_state.query_history if q.get('success')]
        if sources:
            source_counts = pd.Series(sources).value_counts()
            
            fig = go.Figure(data=[go.Pie(labels=source_counts.index, values=source_counts.values)])
            fig.update_layout(title="Solution Sources Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Query history table
        st.markdown("### 📝 Query History")
        history_df = pd.DataFrame([
            {
                'Timestamp': q.get('timestamp', ''),
                'Query': q.get('query', '')[:50] + '...' if len(q.get('query', '')) > 50 else q.get('query', ''),
                'Success': '✅' if q.get('success') else '❌',
                'Source': q.get('source', 'unknown'),
                'Feedback': '✅' if q.get('feedback') else '❌'
            }
            for q in st.session_state.query_history
        ])
        
        st.dataframe(history_df, use_container_width=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧮 Mathematical AI Professor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        # Initialize button
        if st.button("Initialize System", key="init_system"):
            if initialize_orchestrator():
                st.rerun()
        
        st.markdown("---")
        
        # System status
        if st.session_state.orchestrator:
            if st.button("Check System Status"):
                display_system_status()
        
        st.markdown("---")
        
        # Navigation
        st.header("📋 Navigation")
        show_analytics = st.checkbox("Show Analytics", False)
        show_feedback_history = st.checkbox("Show Feedback History", False)
        
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.session_state.feedback_mode = False
            st.session_state.current_solution = None
            st.success("History cleared!")
            st.rerun()
    
    # Main content area
    if not st.session_state.orchestrator:
        st.info("👆 Please configure and initialize the system using the sidebar")
        st.markdown("""
        ### Welcome to Mathematical AI Professor! 🎓
        
        This system provides step-by-step mathematical solutions using:
        - 🔍 **Knowledge Base Search** - First checks existing solutions
        - 🌐 **Web Search** - Searches online for solutions if not found
        - 🧠 **AI Solver** - Generates solutions using advanced AI
        - 👥 **Human Feedback** - Incorporates your feedback for improvement
        
        **Getting Started:**
        1. Enter your Google Gemini API key in the sidebar
        2. Click "Initialize System"
        3. Start asking mathematical questions!
        """)
        return
    
    # Feedback mode
    if st.session_state.feedback_mode:
        display_feedback_interface()
        return
    
    # Main query interface
    st.markdown("### 🤔 Ask a Mathematical Question")
    
    # Query input
    query = st.text_area(
        "Enter your mathematical problem:",
        placeholder="e.g., Solve the quadratic equation x² + 5x + 6 = 0",
        height=100
    )
    
    # Process query
    if st.button("🚀 Solve Problem", key="solve_btn"):
        if query.strip():
            with st.spinner("🔍 Searching knowledge base → 🌐 Web search → 🧠 AI solving..."):
                start_time = time.time()
                result = st.session_state.orchestrator.process_query(query)
                end_time = time.time()
                
                # Add to history
                result['query'] = query
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['processing_time'] = f"{end_time - start_time:.2f}s"
                st.session_state.query_history.append(result)
                
                # Display result
                display_solution(result)
        else:
            st.warning("Please enter a mathematical problem")
    
    # Show analytics if requested
    if show_analytics:
        st.markdown("---")
        display_analytics()
    
    # Show feedback history if requested
    if show_feedback_history and st.session_state.orchestrator:
        st.markdown("---")
        st.markdown("### 🔄 Feedback History")
        feedback_history = st.session_state.orchestrator.get_feedback_history()
        
        if feedback_history:
            for i, feedback_item in enumerate(feedback_history):
                with st.expander(f"Feedback {i+1}"):
                    st.write("**Original Solution:**")
                    st.write(feedback_item.get('original', ''))
                    st.write("**Human Feedback:**")
                    st.write(feedback_item.get('feedback', ''))
                    st.write("**Improved Solution:**")
                    st.write(feedback_item.get('improved', ''))
        else:
            st.info("No feedback history available yet")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧮 Mathematical AI Professor - Powered by CrewAI, Gemini AI, and DSPy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()