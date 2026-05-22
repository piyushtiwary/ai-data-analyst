import asyncio
from pprint import pprint

from state import GraphState

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from graphs import (
    preparation_node,
    profiling_node
)


graph_builder = StateGraph(GraphState)

graph_builder.add_node(
    "preparation",
    preparation_node.preparation_node
)

graph_builder.add_node(
    "profiling",
    profiling_node.profiling_node
)

graph_builder.add_edge(
    START,
    "preparation"
)

graph_builder.add_edge(
    "preparation",
    "profiling"
)

graph_builder.add_edge(
    "profiling",
    END
)

graph = graph_builder.compile()


async def main():

    result = await graph.ainvoke(
        {
            "file_path":
            "data/Teen_Mental_Health_Dataset.csv"
        }
    )

    print("\nDataset Shape:")
    pprint(result["shape"])
    
    print("\nStatistical_summary:")
    pprint(result["statistical_summary"]["age"])

    print("\nDtypes:")
    pprint(result["dtypes"])

    print("\nTop Rows:")
    pprint(result["top_rows"])

    print("\nRaw Missing Values:")
    pprint(result["raw_missing_summary"])

    print("\nClean Missing Values:")
    pprint(result["clean_missing_summary"])


if __name__ == "__main__":
    asyncio.run(main())