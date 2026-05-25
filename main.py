import asyncio
from pprint import pprint

from states.root_state import RootState

from langgraph.graph import StateGraph, START, END

from graphs import preparation_node, profiling_node, data_understanding_node, ml_graphs

graph_builder = StateGraph(RootState)

graph_builder.add_node("preparation", preparation_node.preparation_node)

graph_builder.add_node("profiling", profiling_node.profiling_node)

graph_builder.add_node(
    "data_understanding_node", data_understanding_node.data_understanding_node
)

graph_builder.add_node("class_balance", ml_graphs.analyze_class_balance)
graph_builder.add_node("experiment_history", ml_graphs.load_experiment_history)
graph_builder.add_node("data_quality", ml_graphs.data_quality_node)
graph_builder.add_node("statistical_tests", ml_graphs.statistical_tests_node)
graph_builder.add_node("eda_reasoning", ml_graphs.eda_reasoning_node)
graph_builder.add_node("business_insights", ml_graphs.business_insights_node)
graph_builder.add_node("ml_planning", ml_graphs.ml_planning_node)
graph_builder.add_node("train_models", ml_graphs.train_selected_models)
graph_builder.add_node("select_best_model", ml_graphs.select_best_model)
graph_builder.add_node("evidence_validation", ml_graphs.evidence_validation_node)
graph_builder.add_node("validation", ml_graphs.validation_node)
graph_builder.add_node("prepare_retry", ml_graphs.prepare_retry)
graph_builder.add_node("threshold_optimization", ml_graphs.threshold_optimization_node)
graph_builder.add_node("ml_reasoning", ml_graphs.ml_reasoning_node)
graph_builder.add_node("build_ml_report", ml_graphs.build_ml_report)
graph_builder.add_node("build_visualizations", ml_graphs.build_visualizations_node)
graph_builder.add_node("shap_analysis", ml_graphs.shap_analysis_node)
graph_builder.add_node(
    "visualization_narrative", ml_graphs.visualization_narrative_node
)
graph_builder.add_node("generate_pdf_report", ml_graphs.generate_pdf_report_node)


graph_builder.add_edge(START, "preparation")

graph_builder.add_edge("preparation", "profiling")

graph_builder.add_edge("profiling", "data_understanding_node")

graph_builder.add_edge("data_understanding_node", "class_balance")
graph_builder.add_edge("class_balance", "experiment_history")
graph_builder.add_edge("experiment_history", "data_quality")
graph_builder.add_edge("data_quality", "statistical_tests")
graph_builder.add_edge("statistical_tests", "eda_reasoning")
graph_builder.add_edge("eda_reasoning", "business_insights")
graph_builder.add_edge("business_insights", "ml_planning")
graph_builder.add_edge("ml_planning", "train_models")
graph_builder.add_edge("train_models", "select_best_model")
graph_builder.add_edge("select_best_model", "evidence_validation")
graph_builder.add_edge("evidence_validation", "validation")
graph_builder.add_conditional_edges(
    "validation",
    ml_graphs.decide_after_validation,
    {
        "retry": "prepare_retry",
        "proceed": "threshold_optimization",
    },
)
graph_builder.add_edge("prepare_retry", "train_models")
graph_builder.add_edge("threshold_optimization", "ml_reasoning")
graph_builder.add_edge("ml_reasoning", "build_ml_report")
graph_builder.add_edge("build_ml_report", "build_visualizations")
graph_builder.add_edge("build_visualizations", "shap_analysis")
graph_builder.add_edge("shap_analysis", "visualization_narrative")
graph_builder.add_edge("visualization_narrative", "generate_pdf_report")
graph_builder.add_edge("generate_pdf_report", END)

graph = graph_builder.compile()


async def main():

    result = await graph.ainvoke(
        {
            "data": {"file_path": "data/Teen_Mental_Health_Dataset.csv"},
            "ml": {
                "selected_models": [],
                "experiments": [],
            },
            "report": {},
        }
    )

    print("\n========== FINAL GRAPH STATE ==========\n")

    for key, value in result.items():

        print(f"\n--- {key.upper()} ---")

        pprint(value)


if __name__ == "__main__":
    asyncio.run(main())
