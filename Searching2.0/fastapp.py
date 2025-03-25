
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

fastapp = Flask(__name__)

class EcommerceSearch:
    def __init__(self, excel_file):
        print("Loading dataset...")
        self.df = pd.read_excel(excel_file)
        self.df.fillna("", inplace=True)  # Fill NaN for safety
        self.columns = self.df.columns.tolist()

        # Precompute lowercase values for fast vectorized search
        for col in self.columns:
            self.df[f"{col}_lower"] = self.df[col].astype(str).str.lower()
        print(f"Dataset loaded with {len(self.df)} records and optimized for fast search.")

    def search_data(self, user_query, max_results=5, fuzzy_threshold=60):
        query_lower = user_query.lower()

        # Vectorized multi-column search
        match_mask = np.column_stack([
            self.df[f"{col}_lower"].str.contains(query_lower, na=False)
            for col in self.columns
        ]).any(axis=1)

        result_df = self.df[match_mask]

        # If exact/partial match found, return it
        if not result_df.empty:
            message = f"Found {len(result_df)} matching records."
            return self.format_results(result_df.head(max_results), message)

        # Fuzzy fallback if no direct matches
        message = "No exact match found. Showing closest matches based on fuzzy search."
        combined_data = self.df[self.columns].astype(str).agg(' '.join, axis=1).tolist()
        fuzzy_matches = process.extract(user_query, combined_data, scorer=fuzz.WRatio, limit=max_results)

        results = []
        for match_text, score, index in fuzzy_matches:
            if score >= fuzzy_threshold:
                row = self.df.iloc[index]
                results.append(row.to_dict())

        return {"message": message, "results": results}

    def format_results(self, result_df, message):
        results = [row.to_dict() for _, row in result_df.iterrows()]
        return {"message": message, "results": results}

# Initialize search system with your uploaded Excel file
search_system = EcommerceSearch('F:/PAR Solutions/text 2 image model/Searching/data/10turtle Ecommerce Page.xlsx')

@fastapp.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_query = request.form.get('query', '')
        if user_query:
            result = search_system.search_data(user_query)
    return render_template('2index.html', result=result)

@fastapp.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    result = search_system.search_data(user_query)
    return jsonify(result), 200

if __name__ == '__main__':
    fastapp.run(debug=True)

#------------------------------------------------------------------------------------------------TRY 2
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import json
# from rapidfuzz import process, fuzz

# fastapp = Flask(__name__)

# class FalconJSONQA:
#     def __init__(self, json_path):
#         print("Loading service database...")
#         self.df = self.load_json(json_path)
#         # Indexing by 'category' and creating name-based index
#         self.df.set_index('category', inplace=True)
#         self.service_names = self.df['name'].tolist()  # For Fuzzy Matching
#         print("Indexes set and service names cached.")

#     def load_json(self, json_path):
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         df = pd.json_normalize(data)
#         # Precompute lowercase columns for faster search
#         df['category_lower'] = df['category'].str.lower()
#         df['name_lower'] = df['name'].str.lower()
#         print(f"Loaded {len(df)} records from JSON with lowercase pre-computation.")
#         return df

#     def fast_search(self, user_query, max_records=5):
#         query_lower = user_query.lower()

#         # Fast filtering using numpy vectorization on precomputed lowercase columns
#         mask = np.char.find(self.df['category_lower'].values.astype(str), query_lower) >= 0
#         name_mask = np.char.find(self.df['name_lower'].values.astype(str), query_lower) >= 0

#         combined_mask = mask | name_mask
#         search_df = self.df[combined_mask]

#         message = f"Found {len(search_df)} matching records." if not search_df.empty else "No match found."

#         # Prepare results
#         result = []
#         for _, row in search_df.head(max_records).iterrows():
#             record = {
#                 "category": row.get('category', ""),
#                 "sub_category": row.get("sub-category", ""),
#                 "service": row.get("name", ""),
#                 "tags": row.get("tags", [])
#             }
#             result.append(record)

#         return {"message": message, "results": result}

#     def fuzzy_search(self, user_query, limit=5):
#         # Fuzzy search with RapidFuzz
#         matches = process.extract(user_query, self.service_names, scorer=fuzz.WRatio, limit=limit)
#         result = []
#         for match in matches:
#             matched_service = match[0]
#             row = self.df[self.df['name'] == matched_service].iloc[0]
#             record = {
#                 "category": row.get('category', ""),
#                 "sub_category": row.get("sub-category", ""),
#                 "service": row.get("name", ""),
#                 "tags": row.get("tags", [])
#             }
#             result.append(record)

#         message = "Search results based on closest match."
#         return {"message": message, "results": result}

# # Initialize the JSON QA system
# qa_system = FalconJSONQA('F:/PAR Solutions/text 2 image model/Searching/data/output.json')

# @fastapp.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     if request.method == 'POST':
#         user_query = request.form.get('query', '')
#         search_mode = request.form.get('mode', 'fast')
#         if user_query:
#             if search_mode == 'fuzzy':
#                 result = qa_system.fuzzy_search(user_query)
#             else:
#                 result = qa_system.fast_search(user_query)
#     return render_template('index.html', result=result)

# @fastapp.route('/api/search', methods=['POST'])
# def api_search():
#     data = request.get_json()
#     user_query = data.get('query', '')
#     fuzzy = data.get('fuzzy', False)
#     if not user_query:
#         return jsonify({"error": "Query is required."}), 400

#     if fuzzy:
#         result = qa_system.fuzzy_search(user_query)
#     else:
#         result = qa_system.fast_search(user_query)

#     return jsonify(result), 200

# if __name__ == '__main__':
#     fastapp.run(debug=True)


#------------------------------------------------------------------------------------------------TRY 1
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import json

# fastapp = Flask(__name__)

# class FalconJSONQA:
#     def __init__(self, json_path):
#         print("Loading service database...")
#         self.df = self.load_json(json_path)
#         self.df.set_index('category', inplace=True)  # Indexing by category for faster lookup
#         print("Index on 'name (service)' column has been set.")

#     def load_json(self, json_path):
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         df = pd.json_normalize(data)
#         print(f"Loaded {len(df)} records from JSON.")
#         return df

#     def search_context_json(self, user_query, max_records=5):
#         # Search by category index first for better performance
#         search_df = self.df[self.df.index.str.contains(user_query, case=False, na=False)]

#         if search_df.empty:
#             # If no matches, search the entire DataFrame
#             search_df = self.df[self.df.apply(lambda row: user_query.lower() in str(row).lower(), axis=1)]
#             message = "No exact match found. Showing random services."
#         else:
#             message = f"Found {len(search_df)} matching records."

#         # Limit to max_records and create results list
#         result = []
#         for _, row in search_df.head(max_records).iterrows():
#             record = {
#                 "category": row.name,  # 'category' is the index now
#                 "sub_category": row.get("sub-category", ""),
#                 "service": row.get("name", ""),
#                 "tags": row.get("tags", [])
#             }
#             result.append(record)

#         return {"message": message, "results": result}

# # Initialize the JSON QA system
# qa_system = FalconJSONQA('F:/PAR Solutions/text 2 image model/Searching/data/output.json')

# @fastapp.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     if request.method == 'POST':
#         user_query = request.form.get('query', '')
#         if user_query:
#             result = qa_system.search_context_json(user_query)
#     return render_template('index.html', result=result)

# @fastapp.route('/api/search', methods=['POST'])
# def api_search():
#     data = request.get_json()
#     user_query = data.get('query', '')
#     if not user_query:
#         return jsonify({"error": "Query is required."}), 400
#     result = qa_system.search_context_json(user_query)
#     return jsonify(result), 200

# if __name__ == '__main__':
#     fastapp.run(debug=True)
