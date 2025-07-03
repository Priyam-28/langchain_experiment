import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import asyncio
from pydantic import BaseModel, Field

# Set up your API key
os.environ["GOOGLE_API_KEY"] = "your-google-api-key-here"

@dataclass
class RAGState:
    """State object for the RAG workflow"""
    query: str
    documents: List[Document] = None
    retrieved_docs: List[Document] = None
    answer: str = ""
    human_feedback: str = ""
    iteration_count: int = 0
    final_answer: str = ""
    confidence_score: float = 0.0
    needs_human_review: bool = False

class RAGSystem:
    def __init__(self):
        # Initialize Gemini LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            max_output_tokens=1024
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Vector store will be initialized after loading documents
        self.vector_store = None
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load and process documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue
                
            docs = loader.load()
            documents.extend(docs)
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        return split_docs
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for RAG with human-in-the-loop"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("evaluate", self.evaluate_answer)
        workflow.add_node("human_review", self.human_review)
        workflow.add_node("refine", self.refine_answer)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edges for human-in-the-loop
        workflow.add_conditional_edges(
            "evaluate",
            self.should_involve_human,
            {
                "human_review": "human_review",
                "end": END
            }
        )
        
        workflow.add_edge("human_review", "refine")
        workflow.add_edge("refine", "evaluate")
        
        return workflow.compile()
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on query"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load documents first.")
        
        # Retrieve top-k relevant documents
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        retrieved_docs = retriever.get_relevant_documents(state.query)
        state.retrieved_docs = retrieved_docs
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        return state
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using Gemini based on retrieved documents"""
        if not state.retrieved_docs:
            state.answer = "No relevant documents found."
            return state
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer based on the context. If the context doesn't contain 
        enough information to answer the question completely, mention what additional information 
        would be helpful.
        
        Answer:
        """)
        
        # Generate answer
        formatted_prompt = prompt.format(context=context, question=state.query)
        response = self.llm.invoke(formatted_prompt)
        
        state.answer = response
        print(f"Generated answer: {response[:200]}...")
        
        return state
    
    def evaluate_answer(self, state: RAGState) -> RAGState:
        """Evaluate the quality of the generated answer"""
        # Create evaluation prompt
        eval_prompt = ChatPromptTemplate.from_template("""
        Evaluate the following answer based on the question and context provided.
        
        Question: {question}
        Answer: {answer}
        Context: {context}
        
        Please provide:
        1. A confidence score between 0.0 and 1.0
        2. Whether human review is needed (true/false)
        3. Brief reasoning for your evaluation
        
        Format your response as:
        Confidence: [score]
        Human Review: [true/false]
        Reasoning: [brief explanation]
        """)
        
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        formatted_prompt = eval_prompt.format(
            question=state.query,
            answer=state.answer,
            context=context
        )
        
        evaluation = self.llm.invoke(formatted_prompt)
        
        # Parse evaluation (simplified parsing)
        lines = evaluation.split('\n')
        confidence_line = [line for line in lines if 'Confidence:' in line]
        human_review_line = [line for line in lines if 'Human Review:' in line]
        
        if confidence_line:
            try:
                confidence_str = confidence_line[0].split(':')[1].strip()
                state.confidence_score = float(confidence_str)
            except:
                state.confidence_score = 0.5
        
        if human_review_line:
            try:
                human_review_str = human_review_line[0].split(':')[1].strip().lower()
                state.needs_human_review = 'true' in human_review_str
            except:
                state.needs_human_review = state.confidence_score < 0.7
        
        # Additional logic for human review
        if state.confidence_score < 0.6 or state.iteration_count > 2:
            state.needs_human_review = True
        
        print(f"Evaluation - Confidence: {state.confidence_score}, Human Review: {state.needs_human_review}")
        
        return state
    
    def should_involve_human(self, state: RAGState) -> str:
        """Determine if human involvement is needed"""
        if state.needs_human_review and state.iteration_count < 3:
            return "human_review"
        else:
            state.final_answer = state.answer
            return "end"
    
    def human_review(self, state: RAGState) -> RAGState:
        """Handle human-in-the-loop review"""
        print("\n" + "="*50)
        print("HUMAN REVIEW REQUESTED")
        print("="*50)
        print(f"Query: {state.query}")
        print(f"Current Answer: {state.answer}")
        print(f"Confidence Score: {state.confidence_score}")
        print("\nRetrieved Documents:")
        for i, doc in enumerate(state.retrieved_docs):
            print(f"Doc {i+1}: {doc.page_content[:200]}...")
        
        print("\nPlease provide feedback:")
        print("1. Type 'approve' to accept the current answer")
        print("2. Type 'reject' to request refinement")
        print("3. Provide specific feedback for improvement")
        
        feedback = input("Your feedback: ").strip()
        
        if feedback.lower() == 'approve':
            state.final_answer = state.answer
            state.needs_human_review = False
        else:
            state.human_feedback = feedback
            state.iteration_count += 1
        
        return state
    
    def refine_answer(self, state: RAGState) -> RAGState:
        """Refine the answer based on human feedback"""
        if not state.human_feedback:
            return state
        
        # Create refinement prompt
        refine_prompt = ChatPromptTemplate.from_template("""
        You previously provided this answer to the question, but received feedback for improvement.
        
        Original Question: {question}
        Previous Answer: {previous_answer}
        Human Feedback: {feedback}
        
        Context:
        {context}
        
        Please provide an improved answer that addresses the human feedback while staying 
        grounded in the provided context.
        
        Improved Answer:
        """)
        
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        formatted_prompt = refine_prompt.format(
            question=state.query,
            previous_answer=state.answer,
            feedback=state.human_feedback,
            context=context
        )
        
        refined_answer = self.llm.invoke(formatted_prompt)
        state.answer = refined_answer
        
        # Reset feedback for next iteration
        state.human_feedback = ""
        
        print(f"Refined answer: {refined_answer[:200]}...")
        
        return state
    
    async def process_query(self, query: str) -> str:
        """Process a query through the RAG workflow"""
        if self.vector_store is None:
            return "Please load documents first using load_documents() method."
        
        # Initialize state
        initial_state = RAGState(query=query)
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        return result.final_answer or result.answer

# Example usage and testing
def main():
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Example document loading (replace with your actual file paths)
    # document_paths = ["document1.pdf", "document2.txt"]
    # rag_system.load_documents(document_paths)
    
    # For demonstration, let's create some sample documents
    sample_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            metadata={"source": "python_intro.txt"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make predictions from data without being explicitly programmed.",
            metadata={"source": "ml_basics.txt"}
        ),
        Document(
            page_content="LangGraph is a framework for building stateful, multi-agent applications with language models. It extends LangChain with graph-based workflows.",
            metadata={"source": "langgraph_guide.txt"}
        )
    ]
    
    # Create vector store from sample documents
    rag_system.vector_store = FAISS.from_documents(sample_docs, rag_system.embeddings)
    
    # Process queries
    queries = [
        "What is Python programming language?",
        "How does machine learning work?",
        "What is LangGraph used for?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print('='*60)
        
        try:
            result = asyncio.run(rag_system.process_query(query))
            print(f"\nFinal Answer: {result}")
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()