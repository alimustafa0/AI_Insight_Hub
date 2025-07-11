import random
import streamlit as st

def _xor(a, b):
    return a != b

def _implies(a, b):
    return (not a) or b

def _iff(a, b):
    return ((a and b) or (not a and not b))

# Callback function for the selectbox to update the text input
def update_expression_from_select():
    selected_key = st.session_state.common_expr_select
    if selected_key != "Custom Expression":
        st.session_state.logic_expression_input_value = common_expressions[selected_key]
    # If "Custom Expression" is chosen, the text input will retain its custom value,
    # allowing the user to type freely after selecting "Custom".

# --- Question Generation Functions ---
def get_math_q_dynamic(max_num, ops):
    try:
        num_terms = random.randint(2, 3) # 2 or 3 terms
        equation_parts = []
        current_answer = 0

        # Start with a number
        a = random.randint(1, max_num)
        equation_parts.append(str(a))
        current_answer = a

        for _ in range(num_terms - 1):
            op = random.choice(ops)
            b = random.randint(1, max_num)
            
            # Handle potential division by zero
            if op in ['/', '//', '%'] and b == 0:
                b = random.randint(1, max_num) # Reroll b if it's zero and operator is division/modulo

            expr_part = f"{op} {b}"
            equation_parts.append(expr_part)
        
        expr = " ".join(equation_parts)
        
        # Use a safe way to evaluate, avoiding direct eval for complex user input
        # Given our limited operators, eval is acceptable here as expr is controlled.
        try:
            answer = eval(expr)
            if isinstance(answer, float):
                answer = round(answer, 2) # Round float answers
        except (ZeroDivisionError, OverflowError):
            # If a division by zero or overflow occurs despite checks, regenerate
            return get_math_q_dynamic(max_num, ops) 
            
        # Generate choices: include numbers around the answer, ensure uniqueness and correctness
        choices_set = set()
        choices_set.add(answer) # Always include the correct answer
        
        # Add distractor choices
        num_choices = 4
        attempts = 0
        while len(choices_set) < num_choices and attempts < 20:
            distractor = answer + random.randint(-max_num // 2, max_num // 2)
            if op in ['/', '//'] and isinstance(answer, int) and isinstance(distractor, int):
                # For division, try to keep distractors as integers or rounded floats
                distractor = round(distractor, 2) if random.random() > 0.5 else int(distractor)
            
            # Ensure distractors are not too far off, and within a reasonable range
            if abs(distractor - answer) > 0 and abs(distractor - answer) < (max_num * 2):
                    choices_set.add(distractor)
            attempts += 1
        
        choices = list(choices_set)
        # Ensure we have exactly num_choices, pad with more random if needed
        while len(choices) < num_choices:
                choices.append(random.randint(min(choices)-10, max(choices)+10))
                choices = list(set(choices)) # Ensure uniqueness after padding

        random.shuffle(choices)
        
        # Ensure final choices list has exactly 4 items and contains the answer
        if len(choices) > 4:
            choices = random.sample(choices, 4)
        if answer not in choices:
            choices[random.randint(0, len(choices) - 1)] = answer
        random.shuffle(choices)

        return f"What is {expr}?", answer, choices
    except Exception as e:
        st.error(f"Error generating math question: {e}. Trying again...")
        return get_math_q_dynamic(max_num, ops) # Regenerate on error
    
def get_python_q():
    questions = [
        ("What is the output of `print(2 ** 3)`?", 8, [6, 8, 9, 10]),
        ("What is the type of `[1, 2, 3]`?", "list", ["dict", "list", "tuple", "set"]),
        ("What does `len(\"hello\")` return?", 5, [4, 5, 6, 7]),
        ("Which keyword is used to define a function in Python?", "def", ["func", "define", "def", "function"]),
        ("Which of these is a Python data type?", "tuple", ["array", "collection", "object", "tuple"]),
        ("What is the result of `10 % 3`?", 1, [0, 1, 2, 3]),
        ("Which method adds an item to the end of a list?", "append()", ["add()", "insert()", "extend()", "append()"]),
        ("What is `None` in Python?", "A null value", ["An empty string", "Zero", "A null value", "Undefined"]),
        ("Which loop is used for iterating over a sequence?", "for", ["while", "do-while", "loop", "for"])
    ]
    return random.choice(questions)

def get_logic_q_dynamic():
    boolean_values = [True, False]
    operators = ["and", "or"]
    
    num_terms = random.randint(2, 3) # Number of boolean terms
    
    # Start with a random boolean value
    expr_parts = [str(random.choice(boolean_values))]
    
    for _ in range(num_terms - 1):
        op = random.choice(operators)
        val = random.choice(boolean_values)
        
        # Randomly add 'not'
        if random.random() < 0.3: # 30% chance to add 'not'
            expr_parts.append(f"{op} not {val}")
        else:
            expr_parts.append(f"{op} {val}")
    
    expr = " ".join(expr_parts)
    
    try:
        answer = eval(expr)
    except Exception as e:
        st.error(f"Error generating logic question: {e}. Trying again...")
        return get_logic_q_dynamic() # Regenerate on error

    # Generate choices
    choices = list(set([True, False, not answer])) # Ensure True, False and opposite of answer are options
    if len(choices) < 4:
        choices.extend([None, "Error"]) # Add generic distractors if needed
    choices = random.sample(choices, k=min(4, len(choices)))
    if answer not in choices:
        choices[random.randint(0, len(choices)-1)] = answer # Ensure answer is present
    random.shuffle(choices)

    return f"What is the result of: `{expr}`?", answer, choices


# NEW: Pre-defined Common Expressions
common_expressions = {
    "Custom Expression": "A and not B", # Default value if user picks custom
    "AND Gate (A and B)": "A and B",
    "OR Gate (A or B)": "A or B",
    "NOT Gate (not A)": "not A",
    "NAND Gate (not (A and B))": "not (A and B)",
    "NOR Gate (not (A or B))": "not (A or B)",
    "XOR Gate (A xor B)": "A xor B",
    "Implication (A -> B)": "A -> B",
    "Biconditional (A <-> B)": "A <-> B",
    "De Morgan's Law (AND)": "not (A and B) <-> (not A or not B)",
    "De Morgan's Law (OR)": "not (A or B) <-> (not A and not B)",
    "Distributive Law": "(A and (B or C)) <-> ((A and B) or (A and C))",
    "Contrapositive": "(A -> B) <-> (not B -> not A)",
    "Commutative Law (OR)": "(A or B) <-> (B or A)"
}

# Enhanced formulas dictionary with descriptions and categories
# Each formula is a dictionary containing its LaTeX, description, and category
formulas = {
    "Ohm's Law": {
        "latex": r"V = I \cdot R",
        "description": "Relates voltage (V), current (I), and resistance (R) in an electrical circuit. Fundamental to electronics.",
        "category": "Physics"
    },
    "Pythagorean Theorem": {
        "latex": r"a^2 + b^2 = c^2",
        "description": "In a right-angled triangle, the square of the hypotenuse (c) is equal to the sum of the squares of the other two sides (a and b).",
        "category": "Mathematics"
    },
    "Einstein's Energy-Mass Equivalence": {
        "latex": r"E = mc^2",
        "description": "States that energy (E) and mass (m) are interchangeable, proportional by the speed of light (c) squared. A cornerstone of modern physics.",
        "category": "Physics"
    },
    "Area of Circle": {
        "latex": r"A = \pi r^2",
        "description": "Calculates the area (A) of a circle given its radius (r).",
        "category": "Mathematics"
    },
    "Quadratic Formula": {
        "latex": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        "description": "Provides the solutions (roots) for a quadratic equation of the form ax² + bx + c = 0.",
        "category": "Mathematics"
    },
    "Derivative of sin(x)": {
        "latex": r"\frac{d}{dx}\sin(x) = \cos(x)",
        "description": "A fundamental rule in calculus, showing that the rate of change of the sine function is the cosine function.",
        "category": "Mathematics"
    },
    "Limit Definition of Derivative": {
        "latex": r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
        "description": "Defines the derivative of a function f(x) as the limit of the slope of the secant line as h approaches zero.",
        "category": "Mathematics"
    },
    "Bayes' Theorem": {
        "latex": r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
        "description": "Describes the probability of an event (A) based on prior knowledge of conditions that might be related to the event (B). Crucial in probability and statistics.",
        "category": "Statistics"
    },
    "Compound Interest": {
        "latex": r"A = P\left(1 + \frac{r}{n}\right)^{nt}",
        "description": "Calculates the future value (A) of an investment given the principal (P), annual interest rate (r), number of times interest is compounded per year (n), and time (t) in years.",
        "category": "Finance"
    },
    "Newton's Second Law of Motion": {
        "latex": r"F = ma",
        "description": "States that the force (F) acting on an object is equal to the mass (m) of that object multiplied by its acceleration (a).",
        "category": "Physics"
    },
    "Work-Energy Theorem": {
        "latex": r"W = \Delta KE = \frac{1}{2}mv_f^2 - \frac{1}{2}mv_i^2",
        "description": "Relates the net work (W) done on an object to the change in its kinetic energy (KE).",
        "category": "Physics"
    },
    "Ideal Gas Law": {
        "latex": r"PV = nRT",
        "description": "Relates the pressure (P), volume (V), amount (n), and temperature (T) of an ideal gas, with R being the ideal gas constant.",
        "category": "Chemistry"
    },
    "Statistical Variance": {
        "latex": r"\sigma^2 = \frac{\sum (x_i - \mu)^2}{N}",
        "description": "Measures how far a set of numbers are spread out from their average value (mean, μ).",
        "category": "Statistics"
    },
    "Binomial Probability": {
        "latex": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}",
        "description": "Calculates the probability of exactly k successes in n Bernoulli trials, where p is the probability of success on a single trial.",
        "category": "Statistics"
    },
    "Fick's Law of Diffusion": {
        "latex": r"J = -D \frac{d\phi}{dx}",
        "description": "Relates the diffusive flux (J) to the concentration gradient (dφ/dx), with D as the diffusion coefficient.",
        "category": "Chemistry"
    },
    "Law of Cosines": {
        "latex": r"c^2 = a^2 + b^2 - 2ab \cos(C)",
        "description": "Relates the lengths of the sides of a triangle to the cosine of one of its angles.",
        "category": "Mathematics"
    },
    "Quadratic Reciprocity Law": {
        "latex": r"\left(\frac{p}{q}\right)\left(\frac{q}{p}\right) = (-1)^{\frac{(p-1)(q-1)}{4}}",
        "description": "A theorem in number theory that provides conditions for the solvability of quadratic congruences.",
        "category": "Mathematics"
    },
    "Maxwell's Equations (Differential Form, free space)": {
        "latex": r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0} \quad (\text{Gauss's Law for Electricity}) \\ \nabla \cdot \mathbf{B} = 0 \quad (\text{Gauss's Law for Magnetism}) \\ \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \quad (\text{Faraday's Law of Induction}) \\ \nabla \times \mathbf{B} = \mu_0 \left(\mathbf{J} + \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right) \quad (\text{Ampère-Maxwell Law})",
        "description": "A set of four partial differential equations that, together with the Lorentz force law, form the foundation of classical electromagnetism, classical optics, and electric circuits.",
        "category": "Physics"
    }
}