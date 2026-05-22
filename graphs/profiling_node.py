from state import GraphState


async def profiling_node(state: GraphState):

	df = state["clean_df"]
	
	state['columns'] = df.columns.tolist()
	state['dtypes'] = df.dtypes.astype(str).to_dict()
	state['shape'] = df.shape

	state["top_rows"] = (
		df.head(5)
		  .to_dict(orient="records")
	)

	state["bottom_rows"] = (
		df.tail(5)
		  .to_dict(orient="records")
	)

	state["clean_missing_summary"] = (
		df.isnull()
		  .sum()
		  .to_dict()
	)

	state["statistical_summary"] = (
		df.describe(include="all")
		  .fillna("N/A")
		  .to_dict()
	)

	return state