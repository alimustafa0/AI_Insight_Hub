# Final Improved `6_Learning_Tools.py` with all feedback applied

import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import ast
import operator
import random
import pandas as pd
import re
import inspect
import astunparse # This import might not be necessary if ast.unparse is used directly
import plotly.graph_objects as go
from sympy.abc import x
from collections import defaultdict
from modules.utils import _xor, _implies, _iff, update_expression_from_select, get_math_q_dynamic, get_python_q, get_logic_q_dynamic, common_expressions, formulas

st.set_page_config(layout="wide")
st.title("🧠 Teaching & Learning Tools")
st.markdown("A full-featured educational lab for visualizing, explaining, and testing concepts in **Math, Logic, and Programming**.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Math Visualizer", "🔄 Code Explainer", "🧠 Logic Tools", "📜 Formula Library", "❓ Concept Quiz"])

# ------------------------- TAB 1: Math Visualizer -------------------------
with tab1:
    st.subheader("📊 Math Function Visualizer")
    st.markdown("Enter a mathematical expression in terms of `x`. You can use functions like `sin()`, `cos()`, `tan()`, `log()`, `exp()`, `sqrt()`, `abs()`.")
    st.markdown("---") # Separator for better visual organization

    # Initialize session state for inputs to enable clearing
    if "math_expr_input" not in st.session_state:
        st.session_state.math_expr_input = "sin(x)/x"
    if "x_min_input" not in st.session_state:
        st.session_state.x_min_input = -10.0
    if "x_max_input" not in st.session_state:
        st.session_state.x_max_input = 10.0
    if "num_points_input" not in st.session_state:
        st.session_state.num_points_input = 400

    # User inputs for function and plot range/points
    math_expr = st.text_input("Enter a function in x:", value=st.session_state.math_expr_input, key="math_expr_actual")

    col_min, col_max, col_points = st.columns(3)
    with col_min:
        x_min = st.number_input("X-axis Min:", value=st.session_state.x_min_input, step=0.5, key="x_min_actual")
    with col_max:
        x_max = st.number_input("X-axis Max:", value=st.session_state.x_max_input, step=0.5, key="x_max_actual")
    with col_points:
        num_points = st.slider("Number of Plot Points:", min_value=50, max_value=2000, value=st.session_state.num_points_input, step=50, key="num_points_actual")

    # Buttons for drawing and clearing
    col_draw_clear_1, col_draw_clear_2, _ = st.columns([1, 1, 3])
    with col_draw_clear_1:
        draw_button = st.button("📈 Plot Function", key="plot_function_button")
    with col_draw_clear_2:
        if st.button("🧹 Clear Plot & Inputs", key="clear_plot_button"):
            st.session_state.math_expr_input = "sin(x)/x" # Reset to default
            st.session_state.x_min_input = -10.0
            st.session_state.x_max_input = 10.0
            st.session_state.num_points_input = 400
            st.rerun() # Trigger a rerun to clear inputs and plot

    # Plotting logic
    if draw_button and math_expr:
        if x_min >= x_max:
            st.error("Error: X-axis Min must be less than X-axis Max.")
        else:
            with st.spinner("Generating plot..."):
                try:
                    # Parse the expression using SymPy
                    expr = sp.sympify(math_expr)

                    st.markdown("#### Rendered Expression (LaTeX):")
                    st.latex(sp.latex(expr)) # Display LaTeX representation

                    # Create a numerical function from the symbolic expression
                    f = sp.lambdify(x, expr, modules=['numpy'])

                    # Generate x values
                    x_vals = np.linspace(x_min, x_max, num_points)

                    # Evaluate y values, handling potential domain errors
                    # Use a vectorized operation and filter out NaNs/Infs
                    y_vals = f(x_vals)
                    
                    # Handle cases where y_vals might contain non-finite numbers (e.g., log(0), 1/0)
                    # Replace inf with NaN for consistent plotting. Plotly handles NaNs by breaking the line.
                    y_vals[np.isinf(y_vals)] = np.nan 

                    # Create Plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'y = {math_expr}'))

                    fig.update_layout(
                        title=f"Plot of y = {math_expr}",
                        xaxis_title="x",
                        yaxis_title="y",
                        hovermode="x unified", # Shows tooltip for all traces at a given x
                        template="plotly_white", # Clean template
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                except sp.SympifyError:
                    st.error("Invalid math expression. Please check your syntax (e.g., `x**2`, `sin(x)`).")
                except NameError as ne:
                    st.error(f"Unknown function or variable: {ne}. Ensure you are using `x` and standard math functions.")
                except TypeError as te:
                    st.error(f"Type error in expression evaluation: {te}. Check for domain issues (e.g., `log(negative_number)`).")
                except Exception as e:
                    st.error(f"An unexpected error occurred during plotting: {e}. Please ensure the expression is valid and well-defined over the chosen range.")

# ------------------------- TAB 2: Code Explainer -------------------------
with tab2:
    st.subheader("🔄 Python Code Explainer")
    st.markdown("Paste your Python code below to get a breakdown of its structure and common constructs.")

    # Initialize session state for text area content to enable clearing
    if "user_code_input" not in st.session_state:
        st.session_state.user_code_input = ""

    # Text area for user code input.
    # The value is linked to session_state to allow programmatic clearing.
    user_code = st.text_area("Paste Python Code:", value=st.session_state.user_code_input, height=250, help="Enter your Python code here.", key="actual_code_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        explain_button = st.button("📖 Explain Code", key="explain_code_button")
    with col2:
        # Clear button logic: update session state and rerun the app to clear the text area
        if st.button("🧹 Clear Code", key="clear_code_button"):
            st.session_state.user_code_input = ""
            # Optionally clear other related inputs if necessary, e.g., for the Logic tab
            if 'logic_expression_input_value' in st.session_state:
                st.session_state.logic_expression_input_value = ""
            st.rerun() # FIX: Replaced st.experimental_rerun() with st.rerun()


    if explain_button and user_code:
        with st.spinner("Analyzing code..."):
            try:
                tree = ast.parse(user_code)
                
                st.markdown("### Code Structure Breakdown:")
                st.markdown("---") # Simple separator

                # explanation_lines = [] # No longer needed as we print directly

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        args = [arg.arg for arg in node.args.args]
                        defaults = [ast.unparse(d) for d in node.args.defaults]
                        args_str = ", ".join(
                            [f"{arg}" + (f"={defaults.pop(0)}" if defaults else "") for arg in args]
                        )
                        st.markdown(
                            f"- 🛠️ **Function Definition:** Identified function `` `{node.name}({args_str})` ``. This block defines a reusable piece of code."
                        )
                    elif isinstance(node, ast.ClassDef):
                        bases = [ast.unparse(b) for b in node.bases]
                        bases_str = f"({', '.join(bases)})" if bases else ""
                        st.markdown(
                            f"- 📦 **Class Definition:** Identified class `` `{node.name}{bases_str}` ``. This defines a new blueprint for creating objects."
                        )
                    elif isinstance(node, ast.Assign):
                        targets = ', '.join([ast.unparse(t) for t in node.targets])
                        value = ast.unparse(node.value)
                        st.markdown(
                            f"- ✍️ **Assignment:** The value `` `{value}` `` is assigned to variable(s) `` `{targets}` ``."
                        )
                    elif isinstance(node, ast.AnnAssign):
                        target = ast.unparse(node.target)
                        annotation = ast.unparse(node.annotation)
                        value = ast.unparse(node.value) if node.value else "not initialized"
                        st.markdown(
                            f"- ✍️ **Annotated Assignment:** Variable `` `{target}` `` is declared with type hint `` `{annotation}` `` and initialized with `` `{value}` ``."
                        )
                    elif isinstance(node, ast.AugAssign):
                        target = ast.unparse(node.target)
                        op_map = {
                            'Add': '+', 'Sub': '-', 'Mult': '*', 'Div': '/',
                            'FloorDiv': '//', 'Mod': '%', 'Pow': '**',
                            'LShift': '<<', 'RShift': '>>', 'BitOr': '|',
                            'BitXor': '^', 'BitAnd': '&'
                        }
                        op_name = type(node.op).__name__
                        op_symbol = op_map.get(op_name, op_name) # Get symbol or name if not found
                        value = ast.unparse(node.value)
                        st.markdown(
                            f"- 📝 **Augmented Assignment:** Variable `` `{target}` `` is modified by `` `{op_symbol} {value}` ``. e.g., `` `{target} {op_symbol}= {value}` ``."
                        )
                    elif isinstance(node, ast.If):
                        test = ast.unparse(node.test)
                        st.markdown(
                            f"- 🕵️ **Conditional Statement:** An `if` statement checks the condition `` `{test}` ``. Code blocks will execute based on its truthiness."
                        )
                    elif isinstance(node, ast.For):
                        target = ast.unparse(node.target)
                        iter_ = ast.unparse(node.iter)
                        st.markdown(
                            f"- ⚙️ **For Loop:** A `for` loop iterates `` `{target}` `` over the elements in `` `{iter_}` ``."
                        )
                    elif isinstance(node, ast.While):
                        test = ast.unparse(node.test)
                        st.markdown(
                            f"- 🔁 **While Loop:** A `while` loop repeatedly executes as long as the condition `` `{test}` `` is true."
                        )
                    elif isinstance(node, ast.Call):
                        func = ast.unparse(node.func)
                        args = [ast.unparse(arg) for arg in node.args]
                        keywords = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
                        args_str = ", ".join(args + keywords)
                        st.markdown(
                            f"- 📞 **Function Call:** A call is made to function `` `{func}({args_str})` ``."
                        )
                    elif isinstance(node, ast.Import):
                        names = ', '.join([n.name + (f" as {n.asname}" if n.asname else "") for n in node.names])
                        st.markdown(
                            f"- 📥 **Import Statement:** Module(s) `` `{names}` `` are imported."
                        )
                    elif isinstance(node, ast.ImportFrom):
                        names = ', '.join([n.name + (f" as {n.asname}" if n.asname else "") for n in node.names])
                        module = node.module if node.module else "current package"
                        st.markdown(
                            f"- 📤 **From Import Statement:** Specific names `` `{names}` `` are imported from module `` `{module}` ``."
                        )
                    elif isinstance(node, ast.Return):
                        value = ast.unparse(node.value) if node.value else "None"
                        st.markdown(
                            f"- ↩️ **Return Statement:** The function returns the value `` `{value}` ``."
                        )
                    elif isinstance(node, ast.Expr):
                        # Expression Statement (e.g., standalone function call, literal)
                        # Filter out docstrings which are also Expr nodes
                        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and
                                hasattr(node, 'col_offset') and node.col_offset == 0): # rudimentary docstring check
                            value = ast.unparse(node.value)
                            st.markdown(
                                f"- ⚡ **Expression:** A standalone expression `` `{value}` `` is evaluated (e.g., a function call, a literal)."
                            )
                    elif isinstance(node, ast.Delete):
                        targets = ', '.join([ast.unparse(t) for t in node.targets])
                        st.markdown(
                            f"- 🗑️ **Delete Statement:** Variables/elements `` `{targets}` `` are deleted."
                        )
                    # FIX: Removed the specific check for ast.Finally
                    # The 'finally' block statements are still parsed and explained by ast.walk
                    elif isinstance(node, ast.Try):
                        st.markdown(
                            "- 🛡️ **Try Block:** Code within this block is attempted; errors are caught by `except` blocks and `finally` blocks ensure cleanup."
                        )
                    elif isinstance(node, ast.ExceptHandler):
                        type_ = ast.unparse(node.type) if node.type else "any"
                        name_ = f" as {node.name}" if node.name else ""
                        st.markdown(
                            f"- ❗ **Except Block:** This block handles exceptions of type `` `{type_}{name_}` ``."
                        )
                    # No explicit check for ast.Finally needed as its body is part of ast.Try's finalbody
                    # and its contents will be visited by ast.walk anyway.
                    elif isinstance(node, ast.With):
                        items = ', '.join([f"{ast.unparse(item.context_expr)}" + (f" as {ast.unparse(item.optional_vars)}" if item.optional_vars else "") for item in node.items])
                        st.markdown(
                            f"- 📖 **With Statement:** Uses context manager(s) `` `{items}` `` for resource management."
                        )
                    elif isinstance(node, ast.Raise):
                        exc = ast.unparse(node.exc) if node.exc else "current exception"
                        st.markdown(
                            f"- 🔥 **Raise Statement:** An exception `` `{exc}` `` is explicitly raised."
                        )
                    elif isinstance(node, ast.Pass):
                        st.markdown("- ➡️ **Pass Statement:** A `pass` statement acts as a placeholder, doing nothing.")
                    elif isinstance(node, ast.Break):
                        st.markdown("- 🛑 **Break Statement:** A `break` statement immediately exits the innermost loop.")
                    elif isinstance(node, ast.Continue):
                        st.markdown("- ⏭️ **Continue Statement:** A `continue` statement skips the rest of the current loop iteration and moves to the next.")

                st.markdown("---") # Simple separator
                # Check if any explanation was actually generated
                # A more robust check might involve tracking if any relevant node types were found.
                # For now, if the parsing succeeded, we assume some explanation *could* have been given.
                # If the code is just an empty line or comments, nothing will be printed.
                # You could add a simple counter for nodes explained if you want a more specific 'no explainable structures' message.

            except SyntaxError as se:
                st.error(f"Syntax Error in Python code: {se}. Please check your code for syntax correctness.")
            except Exception as e:
                st.error(f"An unexpected error occurred during code analysis: {e}. Please ensure the code is valid Python.")

# ------------------------- TAB 3: Logic Tools -------------------------
# Assuming 'tab3' is already defined in your app's st.tabs() setup
# Example: tab1, tab2, tab3, tab4, tab5 = st.tabs([..., "🧠 Logic Tools", ...])

with tab3:
    st.subheader("🧠 Logic Truth Table Generator")
    st.markdown("Enter logic expressions using single-letter **uppercase variables** (e.g., `A`, `B`, `C`).")
    st.markdown("---")
    st.markdown("#### Supported Operators:")
    st.code("and, or, not", language="python")
    st.code("xor (exclusive OR)", language="python")
    st.code("-> (implies)", language="python")
    st.code("<-> (if and only if / iff)", language="python")
    st.markdown("---")
    st.markdown("#### Examples:")
    st.code("A and not B", language="python")
    st.code("(A or B) xor C", language="python")
    st.code("A -> B", language="python")
    st.code("A <-> B", language="python")
    st.code("not (A and B) <-> (not A or not B)", language="python") # De Morgan's Law example



    # Initialize session state variable for the text input's value
    # This is crucial for Streamlit to manage the input's state correctly
    if 'logic_expression_input_value' not in st.session_state:
        st.session_state.logic_expression_input_value = common_expressions["Custom Expression"]



    # Selectbox for quick common expressions
    selected_common_expr_name = st.selectbox(
        "⚡ Quick Select Common Expression:",
        list(common_expressions.keys()),
        key="common_expr_select", # Unique key for this widget
        on_change=update_expression_from_select # Call the callback when selection changes
    )

    # Text input for the logic expression
    # Its value is linked to st.session_state.logic_expression_input_value
    expr = st.text_input(
        "Enter your Logic Expression:",
        value=st.session_state.logic_expression_input_value,
        key="logic_expression_input" # Unique key for this widget
    )

    generate_logic = st.button("📊 Generate Truth Table", key="generate_logic_button")

    if generate_logic:
        # FIX for Issue 2: Use the live value from the text input
        current_expr = expr # Use 'expr' directly from the st.text_input widget

        # Define operand_pattern here, before its first use in the re.sub calls
        operand_pattern = r'((?:not\s*[A-Z])|[A-Z]|\(.*?\))'

        # FIX for Issue 1: Extract variables BEFORE processing the expression for custom operators
        # Find all unique uppercase single-letter variables (A-Z) in the original expression
        vars = sorted(list(set(re.findall(r'\b[A-Z]\b', current_expr))))

        # Apply replacements iteratively until no more changes are made.
        # This is crucial for handling nested custom operators.
        processed_expr = current_expr
        old_expr = ""
        while old_expr != processed_expr:
            old_expr = processed_expr
            # Process custom operators in a specific order to avoid partial matches
            # (e.g., '<->' before '->' because '->' is a substring of '<->')
            processed_expr = re.sub(
                rf'{operand_pattern}\s*<->\s*{operand_pattern}', # pattern: operand <-> operand
                r'_iff(\1, \2)', # replacement: _iff(operand1, operand2)
                processed_expr,
                flags=re.DOTALL # Allow . to match newlines if expressions span lines (unlikely here, but good practice)
            )
            processed_expr = re.sub(
                rf'{operand_pattern}\s*->\s*{operand_pattern}',
                r'_implies(\1, \2)',
                processed_expr,
                flags=re.DOTALL
            )
            processed_expr = re.sub(
                rf'{operand_pattern}\s*xor\s*{operand_pattern}',
                r'_xor(\1, \2)',
                processed_expr,
                flags=re.DOTALL
            )

        try:
            if not vars:
                st.warning("No variables (like A, B, C) detected in the expression. Please ensure you're using uppercase variables. Attempting to evaluate as a fixed boolean expression.")
                # If no variables, try to evaluate the expression directly
                try:
                    # Provide custom functions and an empty __builtins__ for safety
                    result = eval(processed_expr, {'_xor': _xor, '_implies': _implies, '_iff': _iff, '__builtins__': {}}, {})
                    st.write(f"Result: `{result}`")
                except Exception as e:
                    st.error(f"Error evaluating expression without variables: {e}. Please check syntax for fixed boolean expressions.")

            if len(vars) > 5: # Limit for performance and readability (2^5 = 32 rows max)
                st.warning(f"Detected {len(vars)} variables. For optimal performance and readability, truth tables are limited to 5 variables (max 32 rows). Please simplify your expression.")
                raise ValueError("Too many variables detected.")

            st.markdown(f"**Detected variables:** `{', '.join(vars)}`")

            table_data = []
            final_results = [] # To store only the final boolean results for tautology/contradiction check

            # Iterate through all possible truth combinations for the variables
            for i in range(2**len(vars)):
                # Convert integer 'i' to its binary representation, mapping to boolean values
                vals = [(i >> j) & 1 for j in reversed(range(len(vars)))]

                # Create the local environment for eval(): map variable names to boolean values
                env_eval = {var: bool(val) for var, val in zip(vars, vals)}

                # Define the global environment for eval(), including our custom functions
                eval_globals = {
                    '_xor': _xor,
                    '_implies': _implies,
                    '_iff': _iff,
                    'True': True, # Ensure True/False are available
                    'False': False
                }

                try:
                    # Evaluate the processed expression
                    result = eval(processed_expr, eval_globals, env_eval)
                except Exception as eval_err:
                    st.error(f"Error evaluating expression '{current_expr}' (processed to: '{processed_expr}') with variables {env_eval}: {eval_err}. Please double-check your syntax and parentheses in complex parts.")

                # Prepare row for DataFrame display
                row_dict = {var: bool(val) for var, val in zip(vars, vals)} # Show True/False
                row_dict['Result'] = bool(result)
                table_data.append(row_dict)
                final_results.append(bool(result)) # Add to results list for analysis

            # Display the truth table using a DataFrame
            df_table = pd.DataFrame(table_data)

            # Convert boolean columns to custom string representation (✔️/❌) for better readability
            for col in df_table.columns:
                if df_table[col].dtype == 'bool':
                    df_table[col] = df_table[col].apply(lambda x: "✔️" if x else "❌")

            st.dataframe(df_table, use_container_width=True)

            # NEW: Tautology/Contradiction/Contingency Detection
            st.markdown("---")
            st.markdown("#### Expression Analysis:")
            if all(final_results):
                st.success("🎉 **This expression is a Tautology!** (Always True)")
            elif not any(final_results): # 'not any(false_list)' is true if all are False
                st.error("🚨 **This expression is a Contradiction!** (Always False)")
            else:
                st.info("💡 **This expression is a Contingency.** (Can be True or False depending on variable values)")

        except ValueError as ve: # Catch specific ValueErrors (like "Too many variables")
            st.error(f"Input Error: {ve}")
        except SyntaxError:
            st.error("Syntax Error: The expression is not valid. Please ensure correct usage of `and`, `or`, `not`, `xor`, `->`, `<->` and correct variable names.")
        except NameError as ne:
            st.error(f"Name Error: Undefined variable or function used. Please ensure variables are single uppercase letters (A-Z) and operators are valid. Error: {ne}")
        except Exception as e:
            # General catch-all for any other unexpected errors
            st.error(f"An unexpected error occurred: {e}. Please double-check your expression and variable format (e.g., single uppercase letters).")
            
            # ------------------------- TAB 4: Formula Library -------------------------
with tab4:
    with st.container(): # Use a container to clearly separate this tab's content
        st.subheader("📜 Famous Formula Library")

        # Initialize session state for selected formula if not present
        if 'selected_formula_name' not in st.session_state:
            st.session_state.selected_formula_name = list(formulas.keys())[0] # Default to the first formula

        # --- Enhance Search and Filtering ---
        col_search, col_category, col_surprise = st.columns([3, 2, 1])

        with col_search:
            search_query = st.text_input("🔍 Search by Name or Description", "").lower()

        with col_category:
            all_categories = sorted(list(set([details["category"] for details in formulas.values()])))
            selected_categories = st.multiselect("🗂️ Filter by Category", all_categories, default=all_categories)

        with col_surprise:
            st.write("") # Spacer for alignment
            st.write("") # Spacer for alignment
            if st.button("🎲 Surprise Me!"):
                st.session_state.selected_formula_name = random.choice(list(formulas.keys()))


        # Filter formulas based on search query and categories
        filtered_formula_names = []
        for name, details in formulas.items():
            match_search = True
            if search_query:
                if search_query not in name.lower() and search_query not in details["description"].lower():
                    match_search = False
            
            match_category = True
            if selected_categories:
                if details["category"] not in selected_categories:
                    match_category = False

            if match_search and match_category:
                filtered_formula_names.append(name)
        
        # Sort filtered formulas alphabetically for consistent display
        filtered_formula_names = sorted(filtered_formula_names)

        if not filtered_formula_names:
            st.info("No formulas match your current search and category filters. Try broadening your criteria.")
        else:
            # If the currently selected formula is no longer in the filtered list, default to the first one
            if st.session_state.selected_formula_name not in filtered_formula_names:
                st.session_state.selected_formula_name = filtered_formula_names[0]

            # Use the filtered list for the selectbox
            selected_formula_name = st.selectbox(
                "📚 Choose a formula from the list:",
                filtered_formula_names,
                index=filtered_formula_names.index(st.session_state.selected_formula_name) if st.session_state.selected_formula_name in filtered_formula_names else 0,
                key="formula_select_box" # Add a key to avoid duplicate widget errors
            )

            # Update session state if a new formula is selected manually
            if selected_formula_name != st.session_state.selected_formula_name:
                st.session_state.selected_formula_name = selected_formula_name

            # Display the selected formula
            if st.session_state.selected_formula_name:
                selected_formula_details = formulas[st.session_state.selected_formula_name]
                st.markdown(f"### {st.session_state.selected_formula_name}")
                st.latex(selected_formula_details["latex"])
                st.markdown(f"**Category:** {selected_formula_details['category']}")
                st.markdown(f"**Description:** {selected_formula_details['description']}")


# ------------------------- TAB 5: Concept Quiz -------------------------
with tab5:
    st.subheader("❓ Dynamic Quiz Generator")
    st.markdown("Test your knowledge in Math, Python, or Logic!")

    # --- Robust Session State Initialization ---
    # Each session state variable is checked and initialized individually.
    if 'q' not in st.session_state:
        st.session_state.q = None
    if 'ans' not in st.session_state:
        st.session_state.ans = None
    if 'choices' not in st.session_state:
        st.session_state.choices = []
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = 0
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    if 'current_Youtubeed' not in st.session_state:
        st.session_state.current_Youtubeed = False
    if 'selected_answer' not in st.session_state:
        st.session_state.selected_answer = None # To store selected answer across reruns

    # --- Quiz Configuration (Moved from sidebar to main page) ---
    st.markdown("---") # Separator for visual clarity
    st.markdown("#### Quiz Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox("Choose Topic:", ["Math", "Python", "Logic"], key="quiz_topic_main")
    with col2:
        # Placeholder for other general settings if added later
        pass


    # Math-specific settings (now appear dynamically based on topic selection)
    if topic == "Math":
        st.markdown("##### Math Question Settings")
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            math_max_num = st.slider("Max Number for Math Questions:", 10, 100, 10, key="math_max_num_slider_main")
        with col_math2:
            math_ops = st.multiselect("Math Operators:", ['+', '-', '*', '/', '//', '%', '**'], ['+', '-', '*'], key="math_ops_multiselect_main")
            if not math_ops:
                st.warning("Please select at least one operator for Math questions.")
                math_ops = ['+'] # Fallback
    
    st.markdown("---") # Another separator

    # --- Button Logic ---
    col_quiz_btn1, col_quiz_btn2 = st.columns([1,1])

    with col_quiz_btn1:
        if st.button("🎮 New Question", key="new_question_button") or st.session_state.q is None:
            if topic == "Math":
                q, ans, choices = get_math_q_dynamic(math_max_num, math_ops)
            elif topic == "Python":
                q, ans, choices = get_python_q()
            else: # Logic
                q, ans, choices = get_logic_q_dynamic()
            
            st.session_state.q = q
            st.session_state.ans = ans
            st.session_state.choices = choices
            st.session_state.feedback = None # Clear previous feedback
            st.session_state.current_Youtubeed = False
            st.session_state.selected_answer = None # Reset selected answer
            
            st.rerun() # Rerun to display the new question

    with col_quiz_btn2:
        if st.button("🔄 Reset Quiz Score", key="reset_score_button"):
            st.session_state.correct_answers = 0
            st.session_state.total_questions = 0
            st.session_state.q = None # Clear current question
            st.session_state.feedback = None
            st.session_state.current_Youtubeed = False
            st.session_state.selected_answer = None
            st.success("Quiz score reset!")
            st.rerun() # Rerun to refresh the UI


    # --- Display Question and Choices ---
    if st.session_state.q:
        st.markdown(f"**Question {st.session_state.total_questions + 1}:** {st.session_state.q}")
        
        # Use a consistent key for the radio button to avoid issues on rerun
        # The 'index' argument ensures the previously selected answer (if any) is re-selected on rerun.
        try:
            # Try to find the index of the previously selected answer
            current_index = st.session_state.choices.index(st.session_state.selected_answer)
        except ValueError:
            # If selected_answer is not in current choices (e.g., new question), default to None
            current_index = None

        selected_answer_val = st.radio("Your Answer:", st.session_state.choices, key="quiz_radio_options", index=current_index)
        
        # Store the selected answer in session_state immediately
        st.session_state.selected_answer = selected_answer_val

        if not st.session_state.current_Youtubeed:
            if st.button("✅ Submit Answer", key="submit_answer_button"):
                st.session_state.total_questions += 1
                if st.session_state.selected_answer == st.session_state.ans:
                    st.success("Correct! 🎉")
                    st.session_state.correct_answers += 1
                    st.session_state.feedback = "correct"
                else:
                    st.error(f"Incorrect. The correct answer was `` `{st.session_state.ans}` `` 🙁")
                    st.session_state.feedback = "incorrect"
                st.session_state.current_Youtubeed = True
                st.rerun() # Rerun to show feedback and hide submit button
        else:
            # Display feedback if already answered
            if st.session_state.feedback == "correct":
                st.success("Correct! 🎉")
            elif st.session_state.feedback == "incorrect":
                st.error(f"Incorrect. The correct answer was `` `{st.session_state.ans}` `` 🙁")
            
            pass


    # --- Display Score ---
    st.markdown("---")
    st.markdown(f"### Score: {st.session_state.correct_answers} / {st.session_state.total_questions}")
    if st.session_state.total_questions > 0:
        accuracy = (st.session_state.correct_answers / st.session_state.total_questions) * 100
        st.info(f"Accuracy: {accuracy:.2f}%")