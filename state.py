from typing import TypedDict, Any, List, Dict, Tuple


class GraphState(TypedDict):
    """
    Central shared state for the LangGraph workflow.

    Every node/agent in the graph:
    - reads values from this state
    - updates only the fields it is responsible for

    This acts as the single source of truth across the
    entire AI data analysis pipeline.
    """
    # INPUT / INGESTION    
    file_path: str

    # Detected dataset format. 
    file_type: str

    # Original raw dataframe directly loaded from source.
    raw_df: Any

    # Cleaned and processed dataframe after:
    clean_df: Any

    # List of dataset column names.
    columns: List[str]

    # Dictionary mapping column names to datatype strings.
    dtypes: Dict[Any, Any]

    # Shape of dataframe.
    shape: Tuple[int, int]

    # First N rows of dataset converted to dictionaries.
    top_rows: List[Dict[str, Any]]

    # Last N rows of dataset converted to dictionaries.
    bottom_rows: List[Dict[str, Any]]

    # Missing/null value summary for each column brfore cleaning.
    raw_missing_summary: Dict[Any, Any]
    
    # Missing/null value summary for each column after cleaning.
    clean_missing_summary: Dict[Any, Any]

    # Statistical summary of dataset.
    statistical_summary: Dict[str, Any]
    
    # High-level natural language summary of the dataset.
    dataset_summary: str

    # Predicted business domain/category of dataset.
    business_domain: str
    
    # Target Column for ML tasks
    probable_target_column: str
    
    # Problem type of dataset
    problem_type: str

    # Recommended analytics or ML tasks inferred by AI.
    recommended_tasks: List[str]
    
    # Recommended visulization tasks
    visualization_recommendations: List[str]
    
    # Recommendations for ML models
    ml_recommendations: List[str]

    # Statistical and analytical outputs
    analytics_results: Dict[str, Any]

    # Outputs generated from ML models.
    ml_results: Dict[str, Any]

    # List of generated chart file paths.
    charts: List[str]

    # AI-generated business insights
    insights: List[str]

    # AI-generated business/action recommendations.
    recommendations: List[str]

    # Stores validation and quality checks across pipeline.
    validation_results: Dict[str, Any]

    # Path to generated final report.
    report_path: str