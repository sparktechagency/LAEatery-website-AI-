import os
import re
import json
from typing import List, Dict
from pydantic import Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from data_manager import RestaurantDataManager, get_day_part, PREMIUM_ZIPS

# Global conversation history
conversation_history = []

class RestaurantSearchTool(BaseTool):
    name: str = "restaurant_search"
    description: str = "Search for restaurants. Automatically detects if user wants trend analysis or regular recommendations."
    data_manager: RestaurantDataManager

    def __init__(self, data_manager: RestaurantDataManager):
        super().__init__(data_manager=data_manager)
        self.data_manager = data_manager
    
    def _run(self, 
            location: str, 
            query: str = "",  
            cuisine: str = "", 
            price: str = "", 
            meal_type: str = "",  
            max_results: int = 2,
            budget: str = "",
            **kwargs) -> str: 
        global conversation_history
        
        # Get the actual user query from conversation history
        user_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break

        # Check if this is explicitly a trend-related query
        show_trend_analysis = is_trend_query(user_query.lower())
        
        # Auto-generate query if not provided
        if not query:
            query_parts = []
            if cuisine: query_parts.append(cuisine)
            if meal_type: query_parts.append(meal_type)
            query = " ".join(query_parts) or "restaurants"
            
        # Convert budget to price tier if provided
        if budget:
            price = self._convert_budget_to_price(budget)
        
        # Build search query
        search_query = f"{query} {cuisine} {meal_type} {price}".strip()
        
        results = []
        
        # Search local data
        if self.data_manager.vectorstore:
            docs = self.data_manager.vectorstore.similarity_search(
                f"{search_query} {location}", 
                k=10
            )
            
            for doc in docs:
                restaurant = next((r for r in self.data_manager.restaurants_data 
                                if r.get('id') == doc.metadata.get('id')), None)
                if restaurant:
                    # Apply filters
                    if cuisine and not any(cuisine.lower() in cat['title'].lower() 
                                        for cat in restaurant.get('categories', [])):
                        continue
                    if price and restaurant.get('price') != price:
                        continue
                    
                    # Base restaurant data (always included)
                    restaurant_data = {
                        'name': restaurant.get('name'),
                        'categories': ", ".join([cat['title'] for cat in restaurant.get('categories', [])]),
                        'rating': restaurant.get('rating'),
                        'review_count': restaurant.get('review_count'),
                        'price': restaurant.get('price', 'N/A'),
                        'location': ", ".join(restaurant.get('location', {}).get('display_address', [])),
                        'menu_items': doc.metadata.get('menu_items', 'Popular dishes available'),
                        'ai_summary': self.data_manager.generate_ai_summary(restaurant),
                        'url': restaurant.get('url'),
                        'image_url': restaurant.get('image_url'),
                        'phone': restaurant.get('display_phone'),
                        'coordinates': restaurant.get('coordinates', {}),
                        'transactions': restaurant.get('transactions', [])
                    }
                    
                    # Only calculate and add trend analysis for explicit trend queries
                    if show_trend_analysis:
                        hype_score = self.data_manager.predict_hype_score(restaurant)
                        future_score = self.data_manager.predict_future_popularity(restaurant)
                        
                        restaurant_data.update({
                            'hype_score': round(hype_score, 2),
                            'trend_score': round(future_score, 2),
                            'popularity_prediction': 'Rising Star' if future_score > hype_score + 5 else 
                                                'Hot Right Now' if hype_score > 75 else 'Stable'
                        })
                    
                    results.append(restaurant_data)
        
        # Sort appropriately
        if show_trend_analysis:
            # For trend queries, calculate scores for sorting
            for result in results:
                if 'trend_score' not in result:
                    restaurant = next((r for r in self.data_manager.restaurants_data 
                                    if r.get('name') == result['name']), None)
                    if restaurant:
                        result['trend_score'] = self.data_manager.predict_future_popularity(restaurant)
            results.sort(key=lambda x: x.get('trend_score', 0), reverse=True)
        else:
            results.sort(key=lambda x: x.get('rating', 0), reverse=True)
        
        return json.dumps(results[:max_results], indent=2)
    
    def _convert_budget_to_price(self, budget: str) -> str:
            """Convert USD budget to Yelp price tiers (1-4)"""
            try:
                budget_num = float(budget)
                if budget_num <= 10:
                    return '1'  # $
                elif budget_num <= 30:
                    return '2'  # $$
                elif budget_num <= 60:
                    return '3'  # $$$
                else:
                    return '4'  # $$$$
            except ValueError:
                return ''

class RestaurantNameSearchTool(BaseTool):
    name: str = "restaurant_name_search"
    description: str = "Search for a specific restaurant by name in local dataset and provide suggestions if not found"
    data_manager: RestaurantDataManager

    def __init__(self, data_manager: RestaurantDataManager):
        super().__init__(data_manager=data_manager)
        self.data_manager = data_manager

    def _run(self, restaurant_name: str, location: str = "") -> str:
        """Search for specific restaurant by name"""
        global conversation_history

        restaurant_name_lower = restaurant_name.lower()

        # Get the actual user query from conversation history
        user_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break

        # Check if this is explicitly a trend-related query
        show_trend_analysis = is_trend_query(user_query.lower())

        # Search in local dataset
        found_restaurant = None
        for restaurant in self.data_manager.restaurants_data:
            if restaurant_name_lower in restaurant.get('name', '').lower():
                found_restaurant = restaurant
                break
        
        if found_restaurant:
            # Return detailed information for found restaurant
            categories = [cat['title'] for cat in found_restaurant.get('categories', [])]
            
            restaurant_info = {
                'found': True,
                'name': found_restaurant.get('name'),
                'rating': found_restaurant.get('rating'),
                'review_count': found_restaurant.get('review_count'),
                'categories': categories,
                'price': found_restaurant.get('price', 'N/A'),
                'location': ", ".join(found_restaurant.get('location', {}).get('display_address', [])),
                'phone': found_restaurant.get('display_phone'),
                'url': found_restaurant.get('url'),
                'image_url': found_restaurant.get('image_url'),
                'menu_items': self.data_manager._extract_menu_items(found_restaurant),
                'ai_summary': self.data_manager.generate_ai_summary(found_restaurant)
            }

            # Only calculate and add trend analysis for explicit trend queries
            if show_trend_analysis:
                hype_score = self.data_manager.predict_hype_score(found_restaurant)
                future_score = self.data_manager.predict_future_popularity(found_restaurant)
                
                restaurant_info.update({
                    'hype_score': round(hype_score, 2),
                    'trend_score': round(future_score, 2),
                    'popularity_prediction': 'Rising Star' if future_score > hype_score + 5 else 
                                            'Hot Right Now' if hype_score > 75 else 'Stable'
                })
        
            return json.dumps(restaurant_info, indent=2)
        else:
            # Restaurant not found, provide suggestions
            suggestions = self._get_similar_restaurants(restaurant_name, location, user_query)

            result = {
                'found': False,
                'searched_name': restaurant_name,
                'message': f"Restaurant '{restaurant_name}' not found in our local dataset.",
                'suggestions': suggestions
            }
            
            return json.dumps(result, indent=2)

    def _get_similar_restaurants(self, restaurant_name: str, location: str, query: str) -> List[Dict]:
        """Get similar restaurants using vector search"""
        global conversation_history
        
        # Get the actual user query from conversation history
        user_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break

        # Check if this is explicitly a trend-related query
        show_trend_analysis = is_trend_query(user_query.lower()) or is_trend_query(query.lower())

        if not self.data_manager.vectorstore:
            return []
        
        # Search for similar restaurants
        search_query = f"{restaurant_name} {location} similar restaurant"
        docs = self.data_manager.vectorstore.similarity_search(search_query, k=3)
        
        suggestions = []
        for doc in docs:
            restaurant = next((r for r in self.data_manager.restaurants_data 
                            if r.get('id') == doc.metadata.get('id')), None)
            if restaurant:
                suggestions.append({
                    'name': restaurant.get('name'),
                    'categories': ", ".join([cat['title'] for cat in restaurant.get('categories', [])]),
                    'rating': restaurant.get('rating'),
                    'review_count': restaurant.get('review_count'),
                    'price': restaurant.get('price', 'N/A'),
                    'ai_summary': self.data_manager.generate_ai_summary(restaurant),
                    'menu_items': self.data_manager._extract_menu_items(restaurant),
                })
                # Only calculate and add trend analysis for explicit trend queries
                if show_trend_analysis:
                    for suggestion in suggestions:
                        restaurant = next((r for r in self.data_manager.restaurants_data 
                                        if r.get('name') == suggestion['name']), None)
                        if restaurant:
                            hype_score = self.data_manager.predict_hype_score(restaurant)
                            future_score = self.data_manager.predict_future_popularity(restaurant)

                            suggestion.update({
                                'hype_score': round(hype_score, 2),
                                'trend_score': round(future_score, 2),
                                'popularity_prediction': 'Rising Star' if future_score > hype_score + 5 else 
                                                        'Hot Right Now' if hype_score > 75 else 'Stable'
                            })
        return suggestions

class RestaurantDetailsTool(BaseTool):
    name: str = "restaurant_details"
    description: str = "Get comprehensive detailed information about a specific restaurant including menu items and AI analysis"
    data_manager: RestaurantDataManager

    def __init__(self, data_manager: RestaurantDataManager):
        super().__init__(data_manager=data_manager)
        self.data_manager = data_manager
    
    def _run(self, restaurant_name: str, save_selection: bool = False, user_query: str = "") -> str:
        """Get comprehensive restaurant details with AI enhancements"""
        global conversation_history
        
        # Get the actual user query from conversation history
        user_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break

        # Check if this is explicitly a trend-related query
        show_trend_analysis = is_trend_query(user_query.lower())

        restaurant = None
        for r in self.data_manager.restaurants_data:
            if restaurant_name.lower() in r.get('name', '').lower():
                restaurant = r
                break
        
        if not restaurant:
            return f"Restaurant '{restaurant_name}' not found in local database. Try searching first."

        # Save selection
        self.data_manager.save_selection(
            restaurant.get('name'),
            restaurant.get('coordinates', {}),
            user_query,
        )
        
        # Generate comprehensive details
        categories = [cat['title'] for cat in restaurant.get('categories', [])]
        location = restaurant.get('location', {})
        coordinates = restaurant.get('coordinates', {})
        
        # AI-enhanced information
        menu_items = self.data_manager._extract_menu_items(restaurant)
        ai_summary = self.data_manager.generate_ai_summary(restaurant)
        
        # Generate visit recommendations using LLM
        visit_recommendation = self._generate_visit_recommendation(restaurant)
        
        details = {
            'name': restaurant.get('name'),
            'id': restaurant.get('id'),
            'ai_summary': ai_summary,
            'visit_recommendation': visit_recommendation,
            'rating': restaurant.get('rating'),
            'review_count': restaurant.get('review_count'),
            'categories': categories,
            'price': restaurant.get('price', 'N/A'),
            'phone': restaurant.get('display_phone'),
            'url': restaurant.get('url'),
            'image_url': restaurant.get('image_url'),
            'is_closed': restaurant.get('is_closed', False),
            'coordinates': coordinates,
            'address': location.get('display_address', []),
            'city': location.get('city'),
            'state': location.get('state'),
            'zip_code': location.get('zip_code'),
            'transactions': restaurant.get('transactions', []),
            'distance': restaurant.get('distance'),
            'menu_items': menu_items,
            'best_time_to_visit': get_day_part(),
            'attributes': restaurant.get('attributes', {}),
            'business_hours': restaurant.get('business_hours', [])
        }
        
        # Only add future trend information if explicitly requested
        if show_trend_analysis:
            hype_score = self.data_manager.predict_hype_score(restaurant)
            future_trend = self.data_manager.predict_future_popularity(restaurant)
            details['hype_score'] = round(hype_score, 2)
            details['future_trend_score'] = round(future_trend, 2)
            details['popularity_prediction'] = 'Rising' if future_trend > hype_score else 'Stable'
        
        return json.dumps(details, indent=2)
    
    def _generate_visit_recommendation(self, restaurant: Dict) -> str:
        """Generate personalized visit recommendation"""
        categories = [cat['title'] for cat in restaurant.get('categories', [])]
        time_of_day = get_day_part()
        
        prompt = f"""Based on this restaurant information, provide a brief recommendation on when and why to visit:
        
        Restaurant: {restaurant.get('name')}
        Categories: {', '.join(categories)}
        Rating: {restaurant.get('rating')} stars
        Current time: {time_of_day}
        Price: {restaurant.get('price', 'N/A')}
        
        Give a 1-2 sentence recommendation about the best time to visit and what to expect."""
        
        try:
            response = self.data_manager.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Great choice for {time_of_day} dining with excellent {', '.join(categories[:2]).lower()} options."

class RestaurantConcierge:
    def __init__(self):
        self.data_manager = RestaurantDataManager()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.tools = [
            RestaurantSearchTool(self.data_manager),
            RestaurantDetailsTool(self.data_manager),
            RestaurantNameSearchTool(self.data_manager),
            MealPlanningTool(self.data_manager)
        ]
        self.agent = None
        self.conversation_stage = "greeting"  
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the enhanced LangChain agent"""
        system_prompt = """You are an expert restaurant concierge AI assistant. Your role is to help users find perfect restaurants through a natural conversation flow.
        CONVERSATION FLOW:
        1. GREETING: Welcome users warmly and ask about their dining preferences
        2. GATHERING: Collect requirements through targeted questions
        3. RECOMMENDING: Provide tailored suggestions with appropriate details and then create meal plan
        4. DETAILS: Offer comprehensive information when requested
        5. MEAL PLANNING: Create detailed meal plans by gathering requirements one by one
        6. CLOSING: End professionally when conversation concludes

        TREND QUERY HANDLING:
        - ONLY show hype scores and trend analysis for these specific queries:
        * "What are the top trending restaurants in [location]?"
        * "Which places have the highest trend score?"
        * "Compare [restaurant]'s trend score to others"
        * "Has [restaurant] been gaining/losing traction?"
        * Queries containing: "trend", "trending","trend score", "hype score", "buzz", "hot spot", "compare trends", "gaining/losing traction"

        - For NORMAL queries, focus on:
        * Restaurant quality and ratings
        * Menu items and cuisine type
        * Location and atmosphere
        * NO hype scores or trend analysis

        MEAL PLANNING PROCESS:
        When users ask for meal plans, gather requirements one by one:
        - Ask for location, occasion, group size, dietary restrictions, budget, cuisine preferences
        - When users mention budget (e.g., $100), convert it to price tiers (1=$, 2=$$, etc.) for searches

        STRICT QUESTIONING RULES:
        - Ask ONLY ONE question per response
        - NEVER ask multiple questions at once
        - After each response, ask for the NEXT piece of information needed
        - Maintain this strict sequence of questions:
        1. Location (city/neighborhood)
        2. Occasion (special event/casual dining)
        3. Cuisine preferences
        4. Budget range
        5. Group size
        6. Dietary restrictions

        STRICT RESPONSE RULES:

        RESPONSE FORMAT for normal recommendations (NO SCORES):
        **🍽️ [Restaurant Name]**
        • **Summary**: [AI-generated summary]
        • **Rating**: ⭐ [X.X] stars ([review_count] reviews)
        • **Categories**: [Cuisine types]
        • **Popular Menu**: [Menu items]
        • **Visit Tip**: [Best time/recommendation]
        • **Contact**: [Phone] | [Link to details]

        RESPONSE FORMAT for trend queries (WITH SCORES):
        **🔥 [Restaurant Name]**
        • **Summary**: [AI-generated summary]
        • **Rating**: ⭐ [X.X] stars ([review_count] reviews)
        • **Hype Score**: 🔥 [XX/100]
        • **Trend Score**: 📈 [XX/100]
        • **Trend Analysis**: [Rising/Stable/Declining with explanation]
        • **Categories**: [Cuisine types]
        • **Popular Menu**: [Menu items]

        MEAL PLAN FORMAT:
        **📅 [Duration] Meal Plan for [Location]**
        **Occasion**: [Special event/casual dining]
        **Group Size**: [Number of people]
        **Budget**: [budget range]

        **☀️ Breakfast Options:**
        1. **[Restaurant]** - [Location]
        - [Specific recommendation reason]
        - Timing: [Suggested time]
        - Highlights: [Menu items]

        [Continue for each meal type...]

        BEHAVIOR GUIDELINES:
        - Ask ONE follow-up question per response
        - Only show scores for explicit trend queries
        - Never ask too many questions at once, only one question question per response
        - For meal planning, gather all requirements before generating plan
        - Always end with a helpful follow-up question
        - Keep responses conversational and engaging
        - When users say "thanks", "goodbye", or similar, ask if there's anything else they'd like to know before ending

        CONVERSATION ENDING SIGNALS:
        Only stop asking follow-up questions when user says:
        - "That's perfect, thank you!"
        - "I'm all set"
        - "No more questions"
        - "Goodbye" or "Thanks, that's all"

        ALWAYS ask thoughtful follow-up questions unless the user indicates they're satisfied."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )
    
    def chat(self, user_input: str) -> str:
        """Process user input with conversation flow management"""
        global conversation_history
        
        # Add to global conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Determine conversation stage
            self._update_conversation_stage(user_input)
            
            # Pass the original input to maintain context
            response = self.agent_executor.invoke({"input": user_input})
            assistant_response = response["output"]
            
            # Add assistant response to global history
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
    
    def _update_conversation_stage(self, user_input: str):
        """Update conversation stage based on user input"""
        lower_input = user_input.lower()
        
        if any(word in lower_input for word in ['goodbye', 'bye', 'thanks', 'thank you', 'that\'s all']):
            self.conversation_stage = "closing"
        elif any(word in lower_input for word in ['details', 'more info', 'tell me more']):
            self.conversation_stage = "details"
        elif "meal plan" in user_input.lower():
            self.conversation_stage = "meal_planning"
        elif self.conversation_stage == "greeting":
            self.conversation_stage = "gathering"
        elif self.conversation_stage == "gathering" and any(word in lower_input for word in ['recommend', 'suggest', 'show me']):
            self.conversation_stage = "recommending"
        elif any(word in lower_input for word in ['trend', 'hype', 'popular']):
            self.conversation_stage = "trend_analysis"

class MealPlanningTool(BaseTool):
    name: str = "meal_planning"
    description: str = "Create personalized meal plans by gathering user requirements step by step"
    data_manager: RestaurantDataManager
    planning_state: Dict = Field(default_factory=dict)

    def __init__(self, data_manager: RestaurantDataManager):
        super().__init__(data_manager=data_manager)
        self.data_manager = data_manager
    
    def _run(self, 
            location: str,
            occasion: str = "",
            dietary_restrictions: str = "",
            budget_range: str = "",
            group_size: str = "",
            time_preference: str = "",
            cuisine_preferences: str = "") -> str:
        
        # Gather requirements step by step
        requirements = {
            'location': location,
            'occasion': occasion,
            'dietary_restrictions': dietary_restrictions,
            'budget_range': budget_range,
            'group_size': group_size,
            'time_preference': time_preference,
            'cuisine_preferences': cuisine_preferences
        }
        
        # Generate comprehensive meal plan
        meal_plan = self._generate_meal_plan(requirements)
        return json.dumps(meal_plan, indent=2)
    
    def _generate_meal_plan(self, requirements):
        """Generate a comprehensive meal plan based on requirements"""
        location = requirements['location']
        
        # Search for different meal types
        breakfast_spots = self._find_meal_spots(location, "breakfast coffee", requirements)
        lunch_spots = self._find_meal_spots(location, "lunch casual", requirements)
        dinner_spots = self._find_meal_spots(location, "dinner restaurant", requirements)
        
        meal_plan = {
            'location': location,
            'occasion': requirements.get('occasion', 'Casual dining'),
            'dietary_notes': requirements.get('dietary_restrictions', 'No restrictions specified'),
            'group_size': requirements.get('group_size', 'Not specified'),
            'meal_schedule': {
                'breakfast': {
                    'time_suggestion': '8:00 AM - 10:00 AM',
                    'restaurants': breakfast_spots[:2],
                    'meal_type': 'Light breakfast & coffee'
                },
                'lunch': {
                    'time_suggestion': '12:00 PM - 2:00 PM', 
                    'restaurants': lunch_spots[:2],
                    'meal_type': 'Casual lunch'
                },
                'dinner': {
                    'time_suggestion': '7:00 PM - 9:00 PM',
                    'restaurants': dinner_spots[:2],
                    'meal_type': 'Main dining experience'
                }
            },
            'additional_recommendations': {
                'transportation_tips': 'Consider ride-sharing between locations',
                'timing_advice': 'Make reservations for dinner spots',
                'backup_options': 'Have 1-2 backup choices for each meal'
            }
        }
        
        return meal_plan
    
    def _find_meal_spots(self, location, meal_type, requirements):
        """Find appropriate restaurants for specific meal type"""
        if not self.data_manager.vectorstore:
            return []
        
        # Build search query with requirements
        dietary = requirements.get('dietary_restrictions', '')
        cuisine = requirements.get('cuisine_preferences', '')
        search_query = f"{meal_type} {dietary} {cuisine} {location}".strip()
        
        docs = self.data_manager.vectorstore.similarity_search(search_query, k=8)
        
        restaurants = []
        for doc in docs:
            restaurant = next((r for r in self.data_manager.restaurants_data 
                            if r.get('id') == doc.metadata.get('id')), None)
            if restaurant:
                # Filter by dietary restrictions
                if dietary and not self._matches_dietary_needs(restaurant, dietary):
                    continue
                
                restaurants.append({
                    'name': restaurant.get('name'),
                    'categories': ", ".join([cat['title'] for cat in restaurant.get('categories', [])]),
                    'rating': restaurant.get('rating'),
                    'review_count': restaurant.get('review_count'),
                    'location': ", ".join(restaurant.get('location', {}).get('display_address', [])),
                    'menu_highlights': doc.metadata.get('menu_items', 'Popular dishes'),
                    'atmosphere': self._get_atmosphere_description(restaurant),
                    'phone': restaurant.get('display_phone'),
                    'url': restaurant.get('url')
                })
        
        # Sort by rating
        restaurants.sort(key=lambda x: x.get('rating', 0), reverse=True)
        return restaurants
    
    def _matches_dietary_needs(self, restaurant, dietary_restrictions):
        """Check if restaurant matches dietary requirements"""
        categories = [cat['title'].lower() for cat in restaurant.get('categories', [])]
        dietary_lower = dietary_restrictions.lower()
        
        # Simple matching logic
        if 'vegetarian' in dietary_lower or 'vegan' in dietary_lower:
            return any(term in ' '.join(categories) for term in ['vegetarian', 'vegan', 'salad', 'healthy'])
        
        if 'gluten' in dietary_lower:
            return any(term in ' '.join(categories) for term in ['gluten', 'healthy', 'salad'])
        
        return True  # Default to include if no specific restrictions
    
    def _get_atmosphere_description(self, restaurant):
        """Generate atmosphere description"""
        categories = [cat['title'] for cat in restaurant.get('categories', [])]
        rating = restaurant.get('rating', 0)
        
        if rating >= 4.5:
            return "Highly rated with excellent atmosphere"
        elif rating >= 4.0:
            return "Great atmosphere and good vibes"
        elif 'casual' in ' '.join(categories).lower():
            return "Casual and relaxed setting"
        else:
            return "Good local spot"

def is_trend_query(query_text):
    """Enhanced trend detection for specific query patterns"""
    trend_patterns = [
        # Direct trend requests
        r'(?i)\btrending?\s+restaurants?\b',
        r'(?i)\btop\s+\d*\s*trending\b',
        r'(?i)\bhighest\s+trend\s+score\b',
        r'(?i)\btrend\s+score\b',
        r'(?i)\bhype\s+score\b',
        
        # Comparison queries
        r'(?i)\bcompare\b.*\btrend\b',
        r'(?i)\bcompare\b.*\bto\s+similar\b',
        
        # Traction/momentum queries
        r'(?i)\b(?:gaining|losing)\s+(?:traction|popularity|momentum)\b',
        r'(?i)\bhas\s+\w+\s+been\s+(?:gaining|losing)\b',
        
        # Hot spots and buzz
        r'(?i)\bhot\s+spots?\b',
        r'(?i)\bbuzz\w*\b',
        r'(?i)\bviral\b',
        r'(?i)\bupcoming\b.*\b(?:spots?|restaurants?)\b',
        
        # Time-based trend queries
        r'(?i)\bthis\s+week\b.*\btrend',
        r'(?i)\brecently\s+popular\b',
        r'(?i)\bnext\s+big\s+thing\b'
    ]
    
    return any(re.search(pattern, query_text) for pattern in trend_patterns)
