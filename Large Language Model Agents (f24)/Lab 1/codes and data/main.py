import os
import sys
import math
import re
from typing import Dict, List
from autogen import ConversableAgent

# Configuration for Groq API
llm_config = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.environ.get("GROQ_API_KEY"),
            "base_url": "https://api.groq.com/openai/v1",
        }
    ],
    "cache_seed": None,
    "temperature": 0.0,
}

# ====================HELPER FUNCTIONS ====================

def normalize(s: str) -> str:
    return s.strip().lower()

def load_restaurant_data() -> Dict[str, List[str]]:
    """Load all restaurant data from file"""
    data_file = os.path.join(os.path.dirname(__file__), "restaurant-data.txt")
    data = {}
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(".", 1)
            if len(parts) < 2:
                continue
            rest_name = parts[0].strip()
            review = parts[1].strip()
            data.setdefault(rest_name, []).append(review)
    
    return data

# ==================== TASK 1: FETCH RESTAURANT DATA ====================

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    """
    Fetch restaurant reviews from restaurant-data.txt.
    Returns dictionary with restaurant name as key and reviews as value.
    """
    all_data = load_restaurant_data()
    
    # Exact match (case-insensitive)
    q = normalize(restaurant_name)
    for k in all_data.keys():
        if q == normalize(k):
            return {k: all_data[k]}
    
    # Substring match
    for k in all_data.keys():
        if q in normalize(k) or normalize(k) in q:
            return {k: all_data[k]}
    
    # Token-based fuzzy match
    q_tokens = set(re.findall(r'[a-z0-9]+', q))
    best = None
    best_score = 0
    for k in all_data.keys():
        k_tokens = set(re.findall(r'[a-z0-9]+', normalize(k)))
        score = len(q_tokens.intersection(k_tokens))
        if score > best_score:
            best_score = score
            best = k
    
    if best and best_score > 0:
        return {best: all_data[best]}
    
    return {}


# ==================== TASK 3: CALCULATE OVERALL SCORE ====================

def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    """
    Calculate the overall score for a restaurant based on food and service scores.
    
    Formula: sum(sqrt(food_score^2 * service_score)) / (N * sqrt(125)) * 10
    """
    if not food_scores or not customer_service_scores or len(food_scores) != len(customer_service_scores):
        return {restaurant_name: 0.0}
    
    N = len(food_scores)
    total = 0.0
    
    for food, service in zip(food_scores, customer_service_scores):
        total += math.sqrt((food ** 2) * service)
    
    denominator = N * math.sqrt(125)
    overall_score = (total / denominator) * 10
    
    return {restaurant_name: round(overall_score, 3)}


# ==================== MAIN FUNCTION WITH AUTOGEN ====================

def main(user_query: str):
    """
    Main function using AutoGen multi-agent architecture.
    
    Architecture follows lab specifications:
    1. Entrypoint Agent - Coordinates workflow
    2. Data Fetch Step - Extract restaurant and fetch reviews
    3. Review Analysis Agent - Extract food/service scores using LLM
    4. Scoring Step - Calculate final overall score
    """
    
    # ==================== STEP 1: DATA FETCH (Agent-coordinated) ====================
    # Entrypoint agent extracts restaurant name
    
    entrypoint_agent = ConversableAgent(
        name="entrypoint_agent",
        system_message=(
            "You are the entrypoint agent. Extract the restaurant name from the user's query. "
            "Respond with ONLY the restaurant name, nothing else."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    
    # Create a simple user proxy for receiving responses
    user_proxy = ConversableAgent(
        name="user_proxy",
        llm_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # Get restaurant name from entrypoint agent
    result = user_proxy.initiate_chat(
        entrypoint_agent,
        message=f"Extract the restaurant name from this query: {user_query}",
        max_turns=1,
    )
    
    restaurant_name = result.summary.strip()
    
    # Fetch the actual data
    fetched_data = fetch_restaurant_data(restaurant_name)
    
    if not fetched_data:
        print(f"Could not find restaurant: {restaurant_name}")
        return
    
    canonical_name, reviews = next(iter(fetched_data.items()))
    
    # ==================== STEP 2: REVIEW ANALYSIS AGENT ====================
    
    review_analysis_agent = ConversableAgent(
        name="review_analysis_agent",
        system_message=(
            "You are a review analysis agent. Analyze restaurant reviews to extract food and service scores.\n\n"
            "SCORING KEYWORDS (use ONLY these):\n"
            "Score 1: awful, horrible, disgusting\n"
            "Score 2: bad, unpleasant, offensive\n"
            "Score 3: average, uninspiring, forgettable\n"
            "Score 4: good, enjoyable, satisfying\n"
            "Score 5: awesome, incredible, amazing\n\n"
            "RULES:\n"
            "- Each review has EXACTLY TWO keywords from the list above\n"
            "- The FIRST keyword found relates to FOOD quality\n"
            "- The SECOND keyword found relates to CUSTOMER SERVICE quality\n"
            "- Find these keywords in order of appearance in the text\n\n"
            "Output format (one line per review, numbered):\n"
            "1: food=X, service=Y\n"
            "2: food=X, service=Y\n"
            "etc.\n\n"
            "Output ONLY the numbered list, nothing else."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    
    # Prepare reviews message
    reviews_text = f"Restaurant: {canonical_name}\n\n"
    for i, review in enumerate(reviews, 1):
        reviews_text += f"Review {i}: {review}\n"
    
    # Get analysis from agent
    analysis_result = user_proxy.initiate_chat(
        review_analysis_agent,
        message=f"Analyze these reviews and extract scores:\n\n{reviews_text}",
        max_turns=1,
    )
    
    # Parse the analysis to extract scores
    analysis_text = analysis_result.summary
    
    food_scores = []
    service_scores = []
    
    # Parse the agent's response to extract scores
    for line in analysis_text.split('\n'):
        # Look for patterns like "1: food=4, service=5" or "food=4, service=5"
        food_match = re.search(r'food[=:\s]+(\d)', line, re.IGNORECASE)
        service_match = re.search(r'service[=:\s]+(\d)', line, re.IGNORECASE)
        
        if food_match and service_match:
            food_scores.append(int(food_match.group(1)))
            service_scores.append(int(service_match.group(1)))
    
    # Fallback: if agent parsing failed, use deterministic keyword matching
    if len(food_scores) != len(reviews):
        keyword_map = {
            "awful": 1, "horrible": 1, "disgusting": 1,
            "bad": 2, "unpleasant": 2, "offensive": 2,
            "average": 3, "uninspiring": 3, "forgettable": 3,
            "good": 4, "enjoyable": 4, "satisfying": 4,
            "awesome": 5, "incredible": 5, "amazing": 5,
        }
        
        food_scores = []
        service_scores = []
        
        for review in reviews:
            review_lower = review.lower()
            found = []
            
            # Find keywords in order of appearance
            for keyword in keyword_map.keys():
                pos = review_lower.find(keyword)
                if pos != -1:
                    found.append((pos, keyword))
            
            found.sort()  # Sort by position
            
            if len(found) >= 2:
                food_scores.append(keyword_map[found[0][1]])
                service_scores.append(keyword_map[found[1][1]])
            elif len(found) == 1:
                food_scores.append(keyword_map[found[0][1]])
                service_scores.append(keyword_map[found[0][1]])
            else:
                food_scores.append(3)
                service_scores.append(3)
    
    # ==================== STEP 3: CALCULATE OVERALL SCORE ====================
    
    result = calculate_overall_score(canonical_name, food_scores, service_scores)
    score = result[canonical_name]
    
    print(f"{canonical_name}: {score:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<restaurant query>'")
        sys.exit(1)
    
    user_query = " ".join(sys.argv[1:])
    main(user_query)
