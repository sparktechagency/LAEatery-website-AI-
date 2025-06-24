import os
import re
import json
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from data_manager import RestaurantDataManager, PREMIUM_ZIPS

# Global conversation history
conversation_history = []

class RestaurantConcierge:
    def __init__(self):
        self.data_manager = RestaurantDataManager()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.4,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        global conversation_history
        self.user_preferences = {}
        self.last_recommendations = []
        self.trend_keywords = [
            'trending', 'trend score', 'hype score', 'hot spot', 'viral', 'top',
            'buzz', 'popular right now', 'gaining traction', 'losing traction',
            'compare trend', 'hottest', 'blowing up', 'next big thing', 'hot',
            'top [0-9]+ trending', 'gen z', 'influencers', 'poppin', 'bougie',
            'hidden gems', 'under the radar', 'gaining traction', 'losing traction'
        ]
        self.user_location = None
        self.gathering_requirements = False  # Track if we're in requirements gathering mode
        self.gathered_requirements = {}  # Store gathered requirements
        self.special_event_keywords = [
            'meal plan', 'special event', 'celebration', 'anniversary', 
            'birthday', 'proposal', 'corporate', 'group dinner', 'breakfast'
            'date night', 'romantic dinner', 'family gathering',
            'business dinner', 'reunion', 'party', 'day', 'launch', 'dinner'
        ]
    
    def chat(self, user_input: str) -> str:
        """Process user input with direct data access for fast responses"""
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Update user location if mentioned
            self._update_user_location(user_input)
            
            # Analyze query intent using LLM
            query_analysis = self._analyze_query(user_input)
            
            # Check if query is insufficient and needs requirements gathering
            if self._is_query_insufficient(query_analysis, user_input):
                response = self._gather_requirements(query_analysis, user_input)
                conversation_history.append({"role": "assistant", "content": response})
                return response
            
            # Process based on query type
            response = self._process_query(query_analysis, user_input)
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error. Let me help you find great restaurants! What type of cuisine are you looking for?"
            conversation_history.append({"role": "assistant", "content": error_response})
            return error_response
    
    def _is_query_insufficient(self, analysis: Dict, user_input: str) -> bool:
        """Check if query lacks sufficient information for good recommendations"""
        # Don't gather requirements for specific query types
        skip_types = ['greeting', 'closing', 'details', 'app_function', 'menu', 
                      'reservation', 'hours', 'vibe', 'wait_time']
        if analysis.get('query_type') in skip_types:
            return False
        
        # If specific restaurant is mentioned, we have enough info
        if analysis.get('restaurant_name'):
            return False
        
        # Check for special event queries
        is_special_event = analysis.get('is_special_event', False)
        if is_special_event:
            # Special events require location, budget, group size and time
            has_location = bool(analysis.get('location')) or bool(self.user_location)
            has_budget = bool(analysis.get('price_range'))
            has_group_size = bool(analysis.get('group_size'))
            has_time = bool(analysis.get('time_context'))
            
            return not (has_location and has_budget and has_group_size and has_time)
        
        # Check for general dinner queries - require location
        if analysis.get('meal_type') in ['dinner', 'supper'] or 'dinner' in user_input.lower():
            has_location = bool(analysis.get('location')) or bool(self.user_location)
            return not has_location
        
        # Check for vague queries
        vague_patterns = [
            r'^(i\'m |im )?hungry$',
            r'^(recommend|suggest|find) (me )?(something|a place|restaurant)$',
            r'^what\'s good\??$',
            r'^where should i (eat|go)\??$',
            r'^food\??$',
            r'^restaurants?\??$',
            r'^help me (find|choose)$',
            r'^any (recommendations|suggestions)\??$',
            r'^what do you recommend\??$'
        ]
        
        user_input_lower = user_input.lower().strip()
        for pattern in vague_patterns:
            if re.match(pattern, user_input_lower):
                return True
        
        # Check if we have minimal information
        has_location = bool(analysis.get('location')) or bool(self.user_location)
        has_cuisine = bool(analysis.get('cuisine'))
        has_meal_type = bool(analysis.get('meal_type'))
        has_price = bool(analysis.get('price_range'))
        has_special_req = bool(analysis.get('special_requirements'))
        
        # For trend queries, we need at least location or cuisine
        if analysis.get('is_trend_query'):
            return not (has_location or has_cuisine)
        
        # For general search, we need at least 2 pieces of information
        info_count = sum([has_location, has_cuisine, has_meal_type, has_price, has_special_req])
        if info_count < 1 and not analysis.get('is_trend_query'):
            return True
        
        return False
    
    def _gather_requirements(self, analysis: Dict, user_input: str) -> str:
        """Generate follow-up questions to gather requirements"""
        # Merge any previously gathered requirements with current analysis
        for key, value in self.gathered_requirements.items():
            if not analysis.get(key) and value:
                analysis[key] = value
        
        # Determine what information we still need
        missing_info = []
        
        # Handle special event requirements
        if analysis.get('is_special_event'):
            if not (analysis.get('location') or self.user_location):
                missing_info.append('location')
            if not analysis.get('price_range'):
                missing_info.append('budget (price range)')
            if not analysis.get('group_size'):
                missing_info.append('group size')
            if not analysis.get('time_context'):
                missing_info.append('date and time')
        # Handle general dinner requirements
        elif analysis.get('meal_type') in ['dinner', 'supper'] or 'dinner' in user_input.lower():
            if not (analysis.get('location') or self.user_location):
                missing_info.append('location')
        # General requirements gathering
        else:
            if not analysis.get('location') and not self.user_location:
                missing_info.append('location')
            if not analysis.get('cuisine'):
                missing_info.append('cuisine')
            if not analysis.get('meal_type'):
                missing_info.append('meal type (breakfast, lunch, dinner)')
            if not analysis.get('price_range'):
                missing_info.append('price range')
        
        # Create a requirements gathering prompt
        if analysis.get('is_special_event'):
            prompt = f"""The user is planning a special event: "{user_input}"
            
            We need these details to create the perfect experience:
            - Location: {analysis.get('location') or self.user_location or 'Not specified'}
            - Budget: {analysis.get('price_range') or 'Not specified'}
            - Group Size: {analysis.get('group_size') or 'Not specified'}
            - Date/Time: {analysis.get('time_context') or 'Not specified'}
            
            Missing: {', '.join(missing_info)}
            
            Generate a friendly, enthusiastic response that:
            1. Acknowledges the special occasion
            2. Asks for the missing information in a conversational way
            3. Provides examples to make it easy for them to respond
            4. Keeps it brief and natural

            Important:
            • Ask only one question at a time for the missing info

            """
        elif analysis.get('meal_type') in ['dinner', 'supper'] or 'dinner' in user_input.lower():
            prompt = f"""The user asked: "{user_input}"
            
            We need their location to find the best dinner spots nearby.
            
            Current location: {analysis.get('location') or self.user_location or 'Not specified'}
            
            Generate a friendly response that:
            1. Acknowledges they're looking for dinner options
            2. Asks for their location or preferred neighborhood
            3. Provides examples of LA neighborhoods
            4. Keeps it conversational
        
            """
        else:
            prompt = f"""The user asked: "{user_input}"
            
            This query is too vague to provide good restaurant recommendations. 
            We need to gather more information about their preferences.
            
            Current information we have:
            - Location: {analysis.get('location') or self.user_location or 'Not specified'}
            - Cuisine: {analysis.get('cuisine') or 'Not specified'}
            - Meal type: {analysis.get('meal_type') or 'Not specified'}
            - Price range: {analysis.get('price_range') or 'Not specified'}
            - Special requirements: {', '.join(analysis.get('special_requirements', [])) or 'None'}
            
            Missing information: {', '.join(missing_info)}
            
            Generate a friendly, conversational response that:
            1. Acknowledges their request enthusiastically
            2. Asks 2-3 specific questions to gather the most important missing information
            3. Provides examples to make it easy for them to respond
            4. Keeps it brief and natural
            
            Important:
            • Ask only one question at a time

            Make it conversational and friendly, not like a form."""
        
        messages = [
            SystemMessage(content="You are a friendly restaurant concierge gathering preferences to make great recommendations."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages).content
        
        # Mark that we're gathering requirements
        self.gathering_requirements = True
        
        # Store what we've gathered so far
        self.gathered_requirements.update({
            k: v for k, v in analysis.items() 
            if v and k in ['location', 'cuisine', 'meal_type', 'price_range', 'special_requirements', 'group_size', 'time_context']
        })
        
        return response
    
    def _update_user_location(self, user_input: str):
        """Update user location if mentioned in the query"""
        location = self._extract_location(user_input)
        if location:
            self.user_location = location
        elif "near me" in user_input.lower() and self.user_location:
            # Keep current location for "near me" queries
            pass
        elif "near me" in user_input.lower() and not self.user_location:
            # Default to central LA if no location set
            self.user_location = "Downtown LA"
    
    def _analyze_query(self, user_input: str) -> Dict:
        """Use LLM to analyze query intent and extract parameters"""
        # If we're gathering requirements, check if this provides the missing info
        context = ""
        if self.gathering_requirements and self.gathered_requirements:
            context = f"\nPrevious gathered requirements: {json.dumps(self.gathered_requirements)}"
            context += "\nThe user might be providing additional preferences in response to our questions."
        
        analysis_prompt = f"""Analyze this restaurant query and extract the following information.
        Return ONLY a JSON object with these fields:
        
        Query: "{user_input}"{context}
        
        Extract:
        {{
            "query_type": "search|details|trending|recommendation|greeting|closing|menu|reservation|hours|vibe|location_based|time_sensitive|comparison|save|report|app_function",
            "is_trend_query": true/false (only true for explicit top/trend/hype/buzz/viral/hot spot/hidden gems queries),
            "is_special_event": true/false (for celebrations, anniversaries, group events, meal plans),
            "location": "extracted location or empty string",
            "cuisine": "extracted cuisine type or empty string",
            "restaurant_name": "specific restaurant name if mentioned or empty string",
            "meal_type": "breakfast|brunch|lunch|dinner|late_night or empty string",
            "price_range": "$|$$|$$$|$$$$ or empty string based on budget mentions",
            "special_requirements": ["vegan", "family_friendly", "romantic", etc],
            "time_context": "now|tonight|weekend|specific_day or empty string",
            "comparison_target": "restaurant to compare against if mentioned",
            "action_needed": "show_menu|make_reservation|check_hours|get_directions|share|report|subscribe|reset or empty string",
            "limit": "number if specified (e.g., 5 for 'top 5') or null",
            "group_size": "number if specified (e.g., 'for 6 people') or null"
        }}
        
        Rules:
        - Set 'is_special_event' to true for: {', '.join(self.special_event_keywords)}
        - For 'hidden gems' or 'under the radar', add 'hidden_gem' to special_requirements
        - For app functionality: 
            'share' = sharing with friend, 
            'report' = report issue, 
            'subscribe' = newsletter, 
            'reset' = reset preferences
        - Extract limit for 'top X' queries
        - For 'near me', use context from previous messages
        - If user mentions budget constraints like "cheap", "affordable", "budget-friendly" set price_range to "$" or "$$"
        - If user mentions "upscale", "fancy", "special occasion" set price_range to "$$$" or "$$$$"
        - Extract group size if mentioned (e.g., "for 4 people", "group of 6")
        """

        analysis_prompt += """
        - For app functionality queries like 'make a reservation', 'report issue', 'share', 
          'subscribe', or 'reset preferences', set query_type to 'app_function'
        - For 'make a reservation', only set action_needed to 'make_reservation' if no specific restaurant is mentioned
        """
        
        messages = [
            SystemMessage(content="You are a query analyzer. Return only valid JSON."),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Clean response and parse JSON
            json_str = response.content.strip()
            json_str = re.sub(r'```json\s*|\s*```', '', json_str)
            analysis = json.loads(json_str)
            
            # Merge with gathered requirements if we're in gathering mode
            if self.gathering_requirements and self.gathered_requirements:
                for key, value in self.gathered_requirements.items():
                    if not analysis.get(key) and value:
                        analysis[key] = value
                
                # If we now have enough info, clear gathering mode
                if analysis.get('location') or self.user_location:
                    if analysis.get('cuisine') or analysis.get('meal_type') or analysis.get('price_range'):
                        self.gathering_requirements = False
                        self.gathered_requirements = {}
            
            # Handle "near me" location
            if "near me" in user_input.lower() and not analysis.get("location") and self.user_location:
                analysis["location"] = self.user_location
                
            return analysis
        except:
            # Fallback analysis
            print("Fallback analysis...")
            return {
                "query_type": "search",
                "is_trend_query": self._is_trend_query(user_input),
                "is_special_event": self._is_special_event(user_input),
                "location": self._extract_location(user_input),
                "cuisine": "",
                "restaurant_name": "",
                "meal_type": "",
                "price_range": "",
                "special_requirements": [],
                "time_context": "",
                "comparison_target": "",
                "action_needed": "",
                "limit": None,
                "group_size": None
            }
    
    def _process_query(self, analysis: Dict, user_input: str) -> str:
        """Process query based on analysis without using tools"""
        query_type = analysis.get("query_type", "search")
        
        # Handle special events first
        if analysis.get("is_special_event", False):
            return self._handle_special_event(analysis, user_input)
            
        # Handle app functionality next
        if query_type == "app_function":
            return self._handle_app_function(analysis, user_input)
            
        # Handle trend queries next
        if analysis.get("is_trend_query") or "trend" in query_type:
            return self._handle_trending_query(analysis, user_input)
        elif query_type == "greeting":
            return self._handle_greeting()
        elif query_type == "closing":
            return self._handle_closing()
        elif query_type == "details" or analysis.get("restaurant_name"):
            return self._handle_restaurant_details(analysis, user_input)
        elif query_type == "comparison":
            return self._handle_comparison(analysis, user_input)
        elif query_type == "time_sensitive":
            return self._handle_time_sensitive(analysis, user_input)
        elif query_type == "location_based":
            return self._handle_location_based(analysis, user_input)
        elif query_type in ["menu", "reservation", "hours", "vibe", "wait_time"]:
            return self._handle_restaurant_info(analysis, user_input, query_type)
        else:
            return self._handle_search(analysis, user_input)
    
    # NEW METHOD: Special event handler
    def _handle_special_event(self, analysis: Dict, user_input: str) -> str:
        """Handle special event meal plan generation"""
        # Get requirements
        location = analysis.get("location") or self.user_location
        price_range = analysis.get("price_range", "$$")
        group_size = analysis.get("group_size", 4)
        time_context = analysis.get("time_context", "tonight")
        cuisine = analysis.get("cuisine", "")
        
        # Find suitable restaurants
        results = self._search_restaurants(
            location=location,
            cuisine=cuisine,
            price=price_range,
            special_requirements=analysis.get("special_requirements", [])
        )
        
        # Filter for group-friendly restaurants
        group_friendly = [
            r for r in results 
            if r.get('attributes', {}).get('RestaurantsGoodForGroups', False)
        ]
        
        # Sort by rating and review count
        group_friendly.sort(key=lambda x: (x.get('rating', 0), x.get('review_count', 0)), reverse=True)
        
        # Take top 3 options
        top_options = group_friendly[:3]
        
        # Generate meal plan using LLM
        return self._generate_special_event_response(top_options, analysis, user_input)
    
    # NEW METHOD: Generate special event response
    def _generate_special_event_response(self, restaurants: List[Dict], 
                                       analysis: Dict, user_input: str) -> str:
        """Generate response for special events with meal plan"""
        # Prepare restaurant info
        restaurant_info = []
        for r in restaurants:
            info = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'review_count': r.get('review_count'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'price': r.get('price', 'N/A'),
                'location': ", ".join(r.get('location', {}).get('display_address', [])),
                'menu_items': self.data_manager._extract_menu_items(r),
                'ai_summary': self.data_manager.generate_ai_summary(r),
                'special_features': []
            }
            
            # Add special features
            if 'romantic' in analysis.get('special_requirements', []):
                info['special_features'].append("Romantic ambiance")
            if 'family_friendly' in analysis.get('special_requirements', []):
                info['special_features'].append("Family-friendly")
            if 'vegan' in analysis.get('special_requirements', []):
                info['special_features'].append("Vegan options")
                
            restaurant_info.append(info)
        
        # Prepare context for meal plan generation
        context = {
            'event_type': "special event",
            'group_size': analysis.get("group_size", 4),
            'price_range': analysis.get("price_range", "$$"),
            'time': analysis.get("time_context", "evening"),
            'special_requests': analysis.get("special_requirements", [])
        }
        
        prompt = f"""Generate a comprehensive special event plan with meal recommendations.
        
        User Query: "{user_input}"
        Event Details:
        - Location: {analysis.get('location') or self.user_location}
        - Group Size: {analysis.get('group_size', 4)} people
        - Budget: {analysis.get('price_range', '$$')}
        - Time: {analysis.get('time_context', 'tonight')}
        - Special Requests: {', '.join(analysis.get('special_requirements', [])) or 'None'}
        
        Top Restaurant Options:
        {json.dumps(restaurant_info, indent=2)}
        
        Your response should include:
        1. A warm introduction acknowledging the special occasion
        2. 3 restaurant options with:
           - Why they're perfect for this event
           - Signature dishes to try
           - Atmosphere description
        3. A curated meal plan for the top recommended restaurant including:
           - Appetizers
           - Main courses
           - Desserts
           - Beverage pairings
        4. Practical tips (reservations, parking, etc.)
        5. A follow-up question to finalize plans
        
        Make it celebratory and engaging!"""
        
        messages = [
            SystemMessage(content="You are a luxury event planner creating memorable dining experiences."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _handle_app_function(self, analysis: Dict, user_input: str) -> str:
        """Dedicated app functionality handler with proper responses"""
        action = analysis.get("action_needed", "")
        
        # Map actions to proper responses
        responses = {
            "make_reservation": "First make sure the restaurant name, then you can book a reservation",
            "report": "Please contract with our Help & Support to report any issues.",
            "subscribe": "Yes! Join our foodie newsletter here: support@leattery.com",
            "share": "Yes! Tap 'Share' to send recommendations via text or social media.",
            "reset": "You can reset your preferences in your profile settings under 'Personalization'"
        }
        
        # Special case for reservation when a restaurant is mentioned
        if "make_reservation" in action and analysis.get("restaurant_name"):
            return self._handle_restaurant_info(analysis, user_input, "reservation")
        
        return responses.get(action, "I can help with restaurant recommendations. What are you in the mood for?")

    def _is_special_event(self, query: str) -> bool:
        """Check if query is for a special event or meal plan"""
        query_lower = query.lower()
        return any(
            re.search(r'\b' + re.escape(keyword) + r'\b', query_lower)
            for keyword in self.special_event_keywords
        )
    
    def _handle_greeting(self) -> str:
        """Generate greeting using LLM"""
        prompt = """Generate a friendly greeting for a restaurant concierge bot in Los Angeles. 
        Be welcoming and ask about their dining preferences. Use an emoji for visual appeal."""
        
        messages = [
            SystemMessage(content="You are a friendly restaurant concierge in Los Angeles."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content

    def _handle_closing(self) -> str:
        """Generate closing using LLM"""
        prompt = """Generate a warm closing message for a restaurant concierge bot. 
        Wish the user an enjoyable meal and invite them to return for more recommendations. 
        Include a food-related emoji."""
        
        messages = [
            SystemMessage(content="You are a friendly restaurant concierge wrapping up a conversation."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _handle_search(self, analysis: Dict, user_input: str) -> str:
        """Handle general restaurant search"""
        # Get relevant restaurants from dataset
        results = self._search_restaurants(
            location=analysis.get("location", ""),
            cuisine=analysis.get("cuisine", ""),
            price=analysis.get("price_range", ""),
            meal_type=analysis.get("meal_type", ""),
            special_requirements=analysis.get("special_requirements", []),
            hidden_gem="hidden_gem" in analysis.get("special_requirements", []),
            exclude_seafood="no seafood" in user_input.lower() or "don't like seafood" in user_input.lower()
        )
        
        # Handle "based on last search"
        if "based on my last search" in user_input.lower() and self.last_recommendations:
            results = self.last_recommendations
        
        # Fallback to vector DB if no exact matches
        if not results:
            fallback_query = f"{analysis.get('cuisine', '')} {analysis.get('location', '')} {analysis.get('meal_type', '')}"
            results = self._vector_db_fallback(fallback_query.strip(), limit=5)
            self.last_recommendations = results
            return self._generate_search_response(results, analysis, user_input, is_fallback=True)
        
        # Apply time-sensitive filters
        if analysis.get("time_context") in ["now", "tonight"]:
            current_hour = datetime.now().hour
            day_of_week = datetime.now().weekday()
            results = [r for r in results if self._is_open_now(r, current_hour, day_of_week)]
        
        # Store last recommendations
        self.last_recommendations = results[:3]
        
        # Generate response using LLM
        return self._generate_search_response(results[:3], analysis, user_input)
    
    def _handle_trending_query(self, analysis: Dict, user_input: str) -> str:
        """Handle trending queries with dataset verification"""
        limit = int(analysis.get("limit", 5)) if analysis.get("limit") else 5
        
        results = self._search_restaurants(
            location=analysis.get("location", ""),
            cuisine=analysis.get("cuisine", ""),
            price=analysis.get("price_range", ""),
            hidden_gem="hidden_gem" in analysis.get("special_requirements", []),
            exclude_seafood="no seafood" in user_input.lower()
        )
        
        # Vector fallback with dataset verification
        if not results:
            results = self._vector_db_fallback(
                f"{analysis.get('cuisine', '')} {analysis.get('location', '')} trending", 
                limit=limit
            )
        
        # Verify all results are in dataset
        valid_results = [
            r for r in results 
            if any(dr['id'] == r['id'] for dr in self.data_manager.restaurants_data)
        ]
        
        # Calculate scores only for valid results
        for restaurant in valid_results:
            restaurant['hype_score'] = self.data_manager.predict_hype_score(restaurant)
            restaurant['trend_score'] = self.data_manager.predict_future_popularity(restaurant)
            restaurant['popularity_prediction'] = self._get_popularity_prediction(
                restaurant['hype_score'], 
                restaurant['trend_score']
            )
        
        # Apply filters
        if "viral" in user_input.lower():
            valid_results = [r for r in valid_results if r.get('hype_score', 0) > 75]
        if "gen z" in user_input.lower():
            valid_results = [r for r in valid_results if r.get('hype_score', 0) > 70 and r.get('rating', 0) > 4.0]
        
        # Apply time-sensitive filters
        if analysis.get("time_context") in ["now", "tonight", "weekend"]:
            current_hour = datetime.now().hour
            day_of_week = datetime.now().weekday()
            valid_results = [r for r in valid_results if self._is_open_now(r, current_hour, day_of_week)]
        
        valid_results.sort(key=lambda x: x.get('trend_score', 0), reverse=True)
        self.last_recommendations = valid_results[:limit]
        
        return self._generate_trending_response(valid_results[:limit], analysis, user_input)
    
    def _handle_restaurant_details(self, analysis: Dict, user_input: str) -> str:
        """Handle specific restaurant queries"""
        restaurant_name = analysis.get("restaurant_name", "")
        
        if not restaurant_name:
            # Try to extract from last recommendations
            if self.last_recommendations:
                restaurant = self.last_recommendations[0]
            else:
                return "Which restaurant would you like to know more about? Please provide the name."
        else:
            # Find restaurant by name
            restaurant = self._find_restaurant_by_name(restaurant_name)
            
            if not restaurant:
                # Find similar restaurants
                similar = self._find_similar_restaurants(restaurant_name)
                return self._generate_not_found_response(restaurant_name, similar)
        
        # Generate detailed response
        return self._generate_details_response(restaurant, analysis, user_input)
    
    def _search_restaurants(self, location: str = "", cuisine: str = "", 
                       price: str = "", meal_type: str = "", 
                       special_requirements: List[str] = None,
                       hidden_gem: bool = False,
                       exclude_seafood: bool = False) -> List[Dict]:
        """Search restaurants in dataset with proper filtering"""
        results = []
        special_requirements = special_requirements or []
        
        for restaurant in self.data_manager.restaurants_data:
            # Filter by location
            if location:
                location_lower = location.lower()
                address = ", ".join(restaurant.get('location', {}).get('display_address', [])).lower()
                if location_lower not in address:
                    continue
                    
            # Filter by cuisine
            if cuisine:
                cuisine_match = False
                for cat in restaurant.get('categories', []):
                    if cuisine.lower() in cat['title'].lower():
                        cuisine_match = True
                        break
                if not cuisine_match:
                    continue
                    
            # Filter by price
            if price and restaurant.get('price') != price:
                continue
            
            # Filter by meal type (using categories)
            if meal_type:
                meal_match = False
                meal_type = meal_type.lower()
                for cat in restaurant.get('categories', []):
                    if meal_type in cat['title'].lower():
                        meal_match = True
                        break
                if not meal_match:
                    continue
                    
            # Check special requirements
            if 'vegan' in special_requirements:
                vegan_match = False
                for cat in restaurant.get('categories', []):
                    if 'vegan' in cat['title'].lower():
                        vegan_match = True
                        break
                if not vegan_match:
                    continue
                    
            if 'family_friendly' in special_requirements:
                if not restaurant.get('attributes', {}).get('GoodForKids', False):
                    continue
                    
            # Hidden gem filter (high rating, low review count)
            if hidden_gem:
                if restaurant.get('rating', 0) < 4.5 or restaurant.get('review_count', 0) > 500:
                    continue
            
            # Exclude seafood if requested
            if exclude_seafood:
                seafood_match = False
                for cat in restaurant.get('categories', []):
                    if 'seafood' in cat['title'].lower():
                        seafood_match = True
                        break
                if seafood_match:
                    continue
            
            # Add restaurant to results if all filters passed
            results.append(restaurant)
        
        # Sort by rating
        results.sort(key=lambda x: x.get('rating', 0), reverse=True)
        
        return results

    def _find_restaurant_by_name(self, name: str) -> Optional[Dict]:
        """Find restaurant by name with improved matching"""
        name_lower = name.lower().strip()
        
        # First try exact match
        for restaurant in self.data_manager.restaurants_data:
            if name_lower == restaurant.get('name', '').lower():
                return restaurant
        
        # Then try partial match
        for restaurant in self.data_manager.restaurants_data:
            if name_lower in restaurant.get('name', '').lower():
                return restaurant
        
        return None

    def _vector_db_fallback(self, query: str, limit: int = 5) -> List[Dict]:
        """Use vector DB when primary search fails with proper mapping"""
        if not self.data_manager.vectorstore:
            return []
        
        # Get extra results to filter
        docs = self.data_manager.vectorstore.similarity_search(query, k=limit*3)
        results = []
        seen_ids = set()
        
        for doc in docs:
            restaurant_id = doc.metadata.get('id')
            if restaurant_id in seen_ids:
                continue
                
            restaurant = next(
                (r for r in self.data_manager.restaurants_data 
                if r.get('id') == restaurant_id),
                None
            )
            if restaurant:
                results.append(restaurant)
                seen_ids.add(restaurant_id)
                if len(results) >= limit:
                    break
        
        return results
    
    def _find_similar_restaurants(self, name: str, limit: int = 3) -> List[Dict]:
        """Find similar restaurants using vector search"""
        if not self.data_manager.vectorstore:
            return []
        
        docs = self.data_manager.vectorstore.similarity_search(name, k=limit)
        similar = []
        
        for doc in docs:
            restaurant = next((r for r in self.data_manager.restaurants_data 
                             if r.get('id') == doc.metadata.get('id')), None)
            if restaurant:
                similar.append(restaurant)
        
        return similar

    def _generate_search_response(self, restaurants: List[Dict], 
                                analysis: Dict, user_input: str,
                                is_fallback: bool = False) -> str:
        """Generate search response using ONLY dataset restaurants"""
        # Prepare restaurant data
        restaurant_info = []
        for r in restaurants:
            # Verify restaurant exists in our dataset
            if not any(res['id'] == r['id'] for res in self.data_manager.restaurants_data):
                continue
                
            info = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'review_count': r.get('review_count'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'price': r.get('price', 'N/A'),
                'location': ", ".join(r.get('location', {}).get('display_address', [])),
                'phone': r.get('display_phone', 'Not available'),
                'url': r.get('url', ''),
                'menu_items': self.data_manager._extract_menu_items(r),
                'ai_summary': self.data_manager.generate_ai_summary(r),
                'is_hidden_gem': "hidden_gem" in analysis.get("special_requirements", [])
            }
            restaurant_info.append(info)
        
        # Strict prompt to prevent hallucination
        prompt = f"""Generate a natural response for this restaurant query.
        
        User Query: "{user_input}"
        {'Note: Showing similar options' if is_fallback else ''}
        Location: {analysis.get('location', 'Los Angeles')}
        
        Restaurants Found:
        {json.dumps(restaurant_info, indent=2)}
        
        STRICT RULES:
        1. Use ONLY the restaurants provided
        2. Do NOT invent or add any restaurants
        3. Do NOT modify restaurant details
        
        FORMAT for each:
        **🍽️ [Name]**
        • **Summary**: [Use provided ai_summary]
        • **Rating**: ⭐ [X.X] stars ([review_count] reviews)
        • **Categories**: [List categories]
        • **Popular Menu**: [List 2-3 menu_items]
        • **Visit Tip**: [Best time/recommendation]
        • **Atmosphere**: [Description of vibe]
        • **Contact**: [phone] | [url]
        
        Add intro, list restaurants, then ONE relevant follow-up question."""
        
        messages = [
            SystemMessage(content="You are a helpful restaurant concierge. STRICTLY use only provided restaurants."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _generate_trending_response(self, restaurants: List[Dict], 
                                  analysis: Dict, user_input: str) -> str:
        """Generate trending response using ONLY dataset restaurants"""
        # Prepare restaurant data with scores - ensure only dataset restaurants are used
        restaurant_info = []
        for r in restaurants:
            # Verify restaurant exists in our dataset
            if not any(res['id'] == r['id'] for res in self.data_manager.restaurants_data):
                continue
                
            info = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'review_count': r.get('review_count'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'price': r.get('price', 'N/A'),
                'hype_score': round(r.get('hype_score', 0), 1),
                'location': ", ".join(r.get('location', {}).get('display_address', [])),
                'trend_score': round(r.get('trend_score', 0), 1),
                'popularity_prediction': r.get('popularity_prediction'),
                'menu_items': self.data_manager._extract_menu_items(r),
                'ai_summary': self.data_manager.generate_ai_summary(r),
                'is_hidden_gem': "hidden_gem" in analysis.get("special_requirements", []),
                'is_viral': "viral" in user_input.lower() or "tiktok" in user_input.lower(),
                'url': r.get('url', ''),
                'phone': r.get('display_phone', 'Not available')
            }
            restaurant_info.append(info)
        
        # Strict prompt to prevent hallucination
        prompt = f"""Generate a response for this TRENDING restaurant query.
        
        User Query: "{user_input}"
        Time Context: {analysis.get('time_context', '')}
        
        Trending Restaurants:
        {json.dumps(restaurant_info, indent=2)}
        
        STRICT RULES:
        1. Use ONLY the restaurants provided in the list above
        2. Do NOT invent or add any restaurants not in the list
        3. Do NOT modify any restaurant details
        4. If no restaurants match, say "No trending spots found matching your criteria"
        
        FORMAT for each restaurant:
        **🔥 [Restaurant Name]**
        • **Summary**: [Use provided ai_summary]
        • **Rating**: ⭐ [X.X] stars ([review_count] reviews)
        • **Hype Score**: 🔥 [XX/100] 
        • **Trend Score**: 📈 [XX/100] 
        • **Trend Analysis**: [Rising/Stable/Declining with explanation]
        {'• **Social Buzz**: Mention TikTok/Instagram if viral' if any(r['is_viral'] for r in restaurant_info) else ''}
        {'• **Hidden Gem**: Mention if applicable' if any(r['is_hidden_gem'] for r in restaurant_info) else ''}
        • **Categories**: [List categories]
        • **Popular Menu**: [List 2-3 menu_items]
        • **Contact**: [phone] | [url]
        
        Start with an engaging intro, list restaurants, end with ONE follow-up question."""
        
        messages = [
            SystemMessage(content="You are a trend-savvy LA restaurant expert. STRICTLY use only provided restaurants."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _generate_details_response(self, restaurant: Dict, 
                                 analysis: Dict, user_input: str) -> str:
        """Generate detailed restaurant information"""
        # Prepare comprehensive details
        details = {
            'name': restaurant.get('name'),
            'rating': restaurant.get('rating'),
            'review_count': restaurant.get('review_count'),
            'categories': [cat['title'] for cat in restaurant.get('categories', [])],
            'price': restaurant.get('price', 'N/A'),
            'location': ", ".join(restaurant.get('location', {}).get('display_address', [])),
            'phone': restaurant.get('display_phone'),
            'url': restaurant.get('url'),
            'hours': restaurant.get('business_hours', []),
            'transactions': restaurant.get('transactions', []),
            'menu_items': self.data_manager._extract_menu_items(restaurant),
            'ai_summary': self.data_manager.generate_ai_summary(restaurant),
            'attributes': restaurant.get('attributes', {})
        }
        
        # Add trend info only if requested
        if analysis.get('is_trend_query'):
            details['hype_score'] = round(self.data_manager.predict_hype_score(restaurant), 1)
            details['trend_score'] = round(self.data_manager.predict_future_popularity(restaurant), 1)
        
        prompt = f"""Generate a detailed response about this restaurant.
        
        User Query: "{user_input}"
        Restaurant Details:
        {json.dumps(details, indent=2)}
        
        Include:
        - Compelling overview
        - Key details (rating, price, location)
        - What makes it special
        - Best dishes to try
        - Practical info (hours, reservations if available)
        - Vibe/atmosphere description
        
        End with ONE helpful follow-up question.
        Keep it conversational and informative."""
        
        messages = [
            SystemMessage(content="You are a knowledgeable restaurant expert providing detailed information."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _generate_not_found_response(self, restaurant_name: str, 
                                   similar: List[Dict]) -> str:
        """Generate response when restaurant not found"""
        # Prepare similar options
        similar_info = []
        for r in similar[:3]:
            info = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'price': r.get('price', 'N/A'),
                'location': ", ".join(r.get('location', {}).get('display_address', [])),
                'ai_summary': self.data_manager.generate_ai_summary(r)
            }
            similar_info.append(info)
        
        prompt = f"""Generate a helpful response when a restaurant isn't found.
        
        User was looking for: "{restaurant_name}"
        
        Similar Options:
        {json.dumps(similar_info, indent=2)}
        
        Format:
        "I couldn't find '{restaurant_name}', but here are similar options you might enjoy:"
        For each option:
        **🍽️ [Name]**
        • Summary: [Brief description]
        • Rating: ⭐ [X.X] stars
        • Categories: [Cuisine types]
        • Location: [Area]
        
        End with: "Would you like more details about any of these?"
        Keep tone friendly and helpful."""
        
        messages = [
            SystemMessage(content="You are a helpful restaurant concierge."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _get_popularity_prediction(self, hype_score: float, trend_score: float) -> str:
        """Get popularity prediction based on scores"""
        if trend_score > hype_score + 5:
            return "Rising Star"
        elif hype_score > 75:
            return "Hot Right Now"
        elif trend_score > 70:
            return "Trending Up"
        elif trend_score < hype_score - 5:
            return "Losing Traction"
        else:
            return "Stable Favorite"
    
    def _is_trend_query(self, query: str) -> bool:
        """Check if query is trend-related"""
        query_lower = query.lower()
        if re.search(r'top\s*\d+', query_lower):
            return True
        return any(
            re.search(r'\b' + re.escape(keyword) + r'\b', query_lower)
            for keyword in self.trend_keywords
        )

    def _extract_location(self, query: str) -> str:
        """Extract location from query"""
        # Common LA neighborhoods
        locations = [
            'silver lake', 'silverlake', 'west hollywood', 'weho', 'downtown', 'dtla',
            'venice', 'santa monica', 'beverly hills', 'hollywood', 'k-town', 'koreatown',
            'culver city', 'pasadena', 'glendale', 'echo park', 'los feliz', 'mid-city',
            'westwood', 'brentwood', 'manhattan beach', 'hermosa beach', 'marina del rey',
            'santa monica', 'west la', 'east la', 'noho', 'sawtelle', 'arts district'
        ]
        
        query_lower = query.lower()
        for loc in locations:
            if loc in query_lower:
                return loc.title()
        
        # Landmark-based locations
        landmarks = {
            'hollywood bowl': 'Hollywood',
            'lax': 'Westchester',
            'ucla': 'Westwood',
            'usc': 'Downtown',
            'staples center': 'Downtown',
            'crypto.com arena': 'Downtown',
            'griffith observatory': 'Los Feliz',
            'santa monica pier': 'Santa Monica',
            'getty center': 'Brentwood',
            'dodger stadium': 'Echo Park',
            'la convention center': 'Downtown'
        }
        
        for landmark, area in landmarks.items():
            if landmark in query_lower:
                return area
        
        return ""
    
    def _handle_comparison(self, analysis: Dict, user_input: str) -> str:
        """Handle comparison queries"""
        target_name = analysis.get("comparison_target", "")
        target = self._find_restaurant_by_name(target_name)
        
        # Vector DB fallback for target
        if not target:
            similar = self._find_similar_restaurants(target_name, limit=1)
            if similar:
                target = similar[0]
            else:
                return f"I couldn't find '{target_name}'. Could you try another restaurant?"
        
        # Add scores if trend query
        if analysis.get("is_trend_query"):
            target['hype_score'] = self.data_manager.predict_hype_score(target)
            target['trend_score'] = self.data_manager.predict_future_popularity(target)
        
        # Find comparable restaurants
        if analysis.get("location"):
            comparable = self._search_restaurants(location=analysis["location"])
        else:
            comparable = self._find_similar_restaurants(target["name"])
        
        # Filter by price if requested
        if "cheaper" in user_input.lower():
            comparable = [r for r in comparable if self._price_level(r.get('price', '$$')) < self._price_level(target.get('price', '$$$'))]
        
        # Add target to comparison list
        restaurants = [target] + [r for r in comparable if r["id"] != target["id"]][:4]
        
        # Add trend scores for all if requested
        if analysis.get("is_trend_query"):
            for r in restaurants:
                r['hype_score'] = self.data_manager.predict_hype_score(r)
                r['trend_score'] = self.data_manager.predict_future_popularity(r)
        
        return self._generate_comparison_response(restaurants, analysis, user_input)
    
    def _price_level(self, price_str: str) -> int:
        """Convert price string to numeric level"""
        return len(price_str) if price_str else 2
    
    def _generate_comparison_response(self, restaurants: List[Dict],
                                    analysis: Dict, user_input: str) -> str:
        """Generate comparison response with optional trend scores"""
        comparison_data = []
        for r in restaurants:
            data = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'review_count': r.get('review_count'),
                'price': r.get('price', 'N/A'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'location': ", ".join(r.get('location', {}).get('display_address', []))
            }
            
            # Add trend metrics if requested
            if analysis.get("is_trend_query"):
                data.update({
                    'hype_score': round(r.get('hype_score', 0), 1),
                    'trend_score': round(r.get('trend_score', 0), 1),
                    'trend_direction': self._get_popularity_prediction(
                        r.get('hype_score', 0), 
                        r.get('trend_score', 0)
                    )
                })
            
            comparison_data.append(data)
        
        prompt = f"""Generate a detailed comparison of these restaurants:
        User Query: "{user_input}"
        
        Restaurants:
        {json.dumps(comparison_data, indent=2)}
        
        Include:
        - Relative strengths/weaknesses
        - Price-to-value comparison
        {"- Trend trajectory analysis" if analysis.get("is_trend_query") else ""}
        - Best use cases (date night, groups, etc)
        - Final recommendation
        
        Use markdown formatting with clear sections."""
        
        messages = [
            SystemMessage(content="You are a restaurant comparison expert."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content

    def _handle_time_sensitive(self, analysis: Dict, user_input: str) -> str:
        """Handle time-sensitive queries"""
        current_hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Determine meal time
        if 'late' in user_input.lower() or current_hour >= 22:
            meal_time = "late_night"
        elif 'brunch' in user_input.lower() or (8 <= current_hour <= 14 and day_of_week >= 5):
            meal_time = "brunch"
        elif 'lunch' in user_input.lower() or (11 <= current_hour <= 15):
            meal_time = "lunch"
        elif 'happy hour' in user_input.lower() or (16 <= current_hour <= 19):
            meal_time = "happy_hour"
        else:
            meal_time = "dinner"
        
        # Filter restaurants by hours
        open_restaurants = []
        for restaurant in self.data_manager.restaurants_data:
            if self._is_open_now(restaurant, current_hour, day_of_week):
                open_restaurants.append(restaurant)
        
        # Apply other filters
        if analysis.get('location'):
            open_restaurants = [r for r in open_restaurants 
                               if analysis['location'].lower() in str(r.get('location', {})).lower()]
        
        # Sort by rating
        open_restaurants.sort(key=lambda x: x.get('rating', 0), reverse=True)
        
        return self._generate_time_sensitive_response(
            open_restaurants[:3], 
            meal_time, 
            analysis, 
            user_input
        )
    
    def _is_open_now(self, restaurant: Dict, current_hour: int, day_of_week: int) -> bool:
        """Check if restaurant is currently open"""
        hours = restaurant.get('business_hours', [])
        if not hours:
            return True  # Assume open if no hours data
        
        for hour_set in hours:
            for day_hours in hour_set.get('open', []):
                if day_hours['day'] == day_of_week:
                    start = int(day_hours['start'][:2])
                    end = int(day_hours['end'][:2])
                    
                    if day_hours['is_overnight']:
                        if current_hour >= start or current_hour < end:
                            return True
                    else:
                        if start <= current_hour < end:
                            return True
        
        return False
    
    def _generate_time_sensitive_response(self, restaurants: List[Dict], 
                                        meal_time: str, analysis: Dict, 
                                        user_input: str) -> str:
        """Generate response for time-sensitive queries"""
        restaurant_info = []
        for r in restaurants[:3]:
            info = {
                'name': r.get('name'),
                'rating': r.get('rating'),
                'categories': [cat['title'] for cat in r.get('categories', [])],
                'price': r.get('price'),
                'location': r.get('location', {}).get('display_address', [])[0] if r.get('location', {}).get('display_address') else 'Location not specified',
                'special_features': []
            }
            
            # Add special features based on meal time
            if meal_time == "late_night":
                info['special_features'].append("Open late")
            elif meal_time == "happy_hour":
                info['special_features'].append("Happy hour specials")
            elif meal_time == "brunch" and 'breakfast_brunch' in str(r.get('categories', [])):
                info['special_features'].append("Great brunch menu")
            
            restaurant_info.append(info)
        
        prompt = f"""Generate a response for this time-sensitive restaurant query.
        
        User Query: "{user_input}"
        Meal Time: {meal_time}
        Current Time Context: {datetime.now().strftime('%A %I:%M %p')}
        
        Open Restaurants:
        {json.dumps(restaurant_info, indent=2)}
        
        Format each restaurant highlighting:
        - Why it's perfect for {meal_time}
        - Any special features or deals
        - Quick bite or leisurely meal
        
        Be specific about timing and availability."""
        
        messages = [
            SystemMessage(content="You are a restaurant expert who knows the best spots for every time of day."),
            HumanMessage(content=prompt)
        ]
        
        return self.llm.invoke(messages).content
    
    def _handle_location_based(self, analysis: Dict, user_input: str) -> str:
        """Handle location-specific queries"""
        location = analysis.get("location", "")
        
        # Handle special location queries
        if 'walking distance' in user_input.lower():
            if self.user_location:
                return f"Here are top restaurants within walking distance in {self.user_location}: [List of Places]"
            else:
                return "To find restaurants within walking distance, I'll need your current location. Which neighborhood are you in right now?"
        
        if 'near' in user_input.lower():
            # Extract landmark
            landmark_match = re.search(r'near (?:the )?(.+?)(?:\?|$)', user_input.lower())
            if landmark_match:
                landmark = landmark_match.group(1)
                return self._find_near_landmark(landmark, analysis)
        
        # Regular location search
        return self._handle_search(analysis, user_input)
    
    def _find_near_landmark(self, landmark: str, analysis: Dict) -> str:
        """Find restaurants near a landmark"""
        # Map common landmarks to areas
        landmark_areas = {
            'hollywood bowl': 'Hollywood',
            'lax': 'Westchester',
            'ucla': 'Westwood',
            'usc': 'Downtown',
            'staples center': 'Downtown',
            'crypto.com arena': 'Downtown',
            'griffith observatory': 'Los Feliz',
            'santa monica pier': 'Santa Monica',
            'getty center': 'Brentwood',
            'dodger stadium': 'Echo Park',
            'venice beach': 'Venice',
            'la convention center': 'Downtown'
        }
        
        area = landmark_areas.get(landmark.lower(), '')
        if area:
            analysis['location'] = area
            return self._handle_search(analysis, f"restaurants near {landmark}")
        else:
            return f"I'm not familiar with restaurants specifically near {landmark}. Could you tell me which neighborhood that's in?"
    
    def _handle_restaurant_info(self, analysis: Dict, user_input: str, 
                              info_type: str) -> str:
        """Handle specific restaurant information requests"""
        restaurant_name = analysis.get('restaurant_name', '')
        
        if not restaurant_name and self.last_recommendations:
            restaurant = self.last_recommendations[0]
        else:
            restaurant = self._find_restaurant_by_name(restaurant_name)
            if not restaurant:
                return f"Which restaurant would you like {info_type} information for?"
        
        if info_type == "app_general":
            return self._handle_app_function(analysis, user_input)
        
        elif info_type == "menu":
            menu_items = self.data_manager._extract_menu_items(restaurant)
            menu_url = restaurant.get('attributes', {}).get('menu_url', restaurant.get('url'))
            
            response = f"**📋 Menu Highlights at {restaurant.get('name')}**\n\n"
            response += f"Popular dishes: {menu_items}\n\n"
            response += f"For the full menu, visit: {menu_url}\n\n"
            response += "Would you like me to tell you more about the restaurant or help with reservations?"
            
        elif info_type == "reservation":
            transactions = restaurant.get('transactions', [])
            phone = restaurant.get('display_phone', 'Not available')
            
            response = f"**📅 Reservations at {restaurant.get('name')}**\n\n"
            if 'restaurant_reservation' in transactions:
                response += "✅ This restaurant accepts reservations!\n\n"
                response += f"You can call them at: {phone}\n"
                response += f"Or visit their Yelp page: {restaurant.get('url')}\n\n"
            else:
                response += "This restaurant typically doesn't take reservations - it's first come, first served.\n"
                response += f"You can call to confirm: {phone}\n\n"
            response += "Would you like suggestions for the best time to visit?"
            
        elif info_type == "hours":
            hours = restaurant.get('business_hours', [])
            response = f"**🕐 Hours for {restaurant.get('name')}**\n\n"
            
            if hours:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for hour_set in hours:
                    for day_hours in hour_set.get('open', []):
                        day = days[day_hours['day']]
                        start = f"{int(day_hours['start'][:2])}:{day_hours['start'][2:]}"
                        end = f"{int(day_hours['end'][:2])}:{day_hours['end'][2:]}"
                        response += f"{day}: {start} - {end}\n"
            else:
                response += "Hours information not available. Please call for current hours.\n"
            
            response += f"\nPhone: {restaurant.get('display_phone', 'Not available')}\n"
            response += "\nWould you like directions or parking information?"
            
        elif info_type == "vibe":
            categories = [cat['title'] for cat in restaurant.get('categories', [])]
            price = restaurant.get('price', 'N/A')
            
            vibe_prompt = f"""Describe the vibe and atmosphere at this restaurant:
            Name: {restaurant.get('name')}
            Categories: {categories}
            Price: {price}
            Rating: {restaurant.get('rating')}
            
            Create a vivid 2-3 sentence description of what to expect in terms of atmosphere, 
            crowd, noise level, and overall dining experience."""
            
            messages = [
                SystemMessage(content="You are a restaurant atmosphere expert."),
                HumanMessage(content=vibe_prompt)
            ]
            
            vibe_description = self.llm.invoke(messages).content
            
            response = f"**✨ The Vibe at {restaurant.get('name')}**\n\n"
            response += f"{vibe_description}\n\n"
            response += "Would you like to know about the best dishes to order here?"
            
        elif info_type == "wait_time":
            response = f"**⏳ Wait Time at {restaurant.get('name')}**\n\n"
            response += "Typical wait times are around 30-45 minutes during peak hours. "
            response += "You can call ahead to check current wait times."
            
        elif info_type == "family_friendly":
            kids_friendly = restaurant.get('attributes', {}).get('GoodForKids', False)
            response = f"**👨‍👩‍👧‍👦 Family Friendly at {restaurant.get('name')}**\n\n"
            if kids_friendly:
                response += "Yes, this restaurant is family-friendly and has a kids menu available!"
            else:
                response += "This restaurant is more suited for adults. For family-friendly options, I can suggest alternatives."
            
        return response