import asyncio
from pprint import pprint

from states.root_state import RootState

from langgraph.graph import StateGraph, START, END

from graphs import preparation_node, profiling_node, data_understanding_node

graph_builder = StateGraph(RootState)

graph_builder.add_node("preparation", preparation_node.preparation_node)

graph_builder.add_node("profiling", profiling_node.profiling_node)

graph_builder.add_node(
    "data_understanding_node", data_understanding_node.data_understanding_node
)


graph_builder.add_edge(START, "preparation")

graph_builder.add_edge("preparation", "profiling")

graph_builder.add_edge("profiling", "data_understanding_node")

graph = graph_builder.compile()


async def main():

    result = await graph.ainvoke(
        {"data": {"file_path": "data/Teen_Mental_Health_Dataset.csv"}}
    )

    print("\n========== FINAL GRAPH STATE ==========\n")

    for key, value in result.items():

        print(f"\n--- {key.upper()} ---")

        pprint(value)


if __name__ == "__main__":
    asyncio.run(main())
