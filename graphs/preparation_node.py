from state import GraphState
import pandas as pd


async def preparation_node(state: GraphState):
	
	file_path = state['file_path']
	
	file_type = file_path.split(".")[-1]
	
	state['file_type'] = file_type
	
	raw_df = None
 
	# Check file type and load dataframe	
	if file_type == 'csv':
		raw_df = pd.read_csv(file_path)
	elif file_type == 'json':
		raw_df = pd.read_json(file_path)
	elif file_type == 'xlsx':
		raw_df = pd.read_excel(file_path)
	
	if raw_df is not None:
		# Update raw data
		state['raw_df'] = raw_df
  
		state['raw_missing_summary'] = (
			raw_df.isnull()
			.sum()
			.to_dict()
		)


		clean_df = raw_df.copy()

		# Normalize Column Names
		clean_df.columns = [
			col.strip().lower().replace(" ", "_")
			for col in clean_df.columns
		]


		# Remove Duplicate Rows
		clean_df = clean_df.drop_duplicates()
		

		# 3. Remove Fully Empty Rows
		clean_df = clean_df.dropna(how="all")


		# Trim String Whitespace
		for col in clean_df.select_dtypes(include="object").columns:
			clean_df[col] = clean_df[col].astype(str).str.strip()

		
		# Fill Missing Numeric Values
		numeric_cols = clean_df.select_dtypes(include="number").columns

		for col in numeric_cols:
			clean_df[col] = clean_df[col].fillna(
				clean_df[col].median()
			)


		# Fill Missing Categorical Values
		categorical_cols = clean_df.select_dtypes(include="object").columns

		for col in categorical_cols:
			clean_df[col] = clean_df[col].fillna("Unknown")


		# Update State
		state["clean_df"] = clean_df

	return state