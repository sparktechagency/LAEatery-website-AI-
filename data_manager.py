import os
import json
import csv
import numpy as np
import pickle
import requests
from datetime import datetime
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


PREMIUM_ZIPS = ['90036', '90026', '90027', '90293', '90402', '90019', '90022', '91401', '90066', '90077', '90046', '91501', '90049', '91020']

YELP_API_KEY = os.getenv("YELP_API_KEY")

def get_day_part():
    hour = datetime.now().hour
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "late night"

class RestaurantDataManager:
    def __init__(self):
        self.restaurants_data = []
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.hype_model = None
        self.trend_model = None
        self.scaler = StandardScaler()
        self.trend_scaler = StandardScaler()
        self.csv_file = "restaurant_selections.csv"
        
        self.models_dir = "saved_models"
        self.vectorstore_path = os.path.join(self.models_dir, "vectorstore")
        self.hype_model_path = os.path.join(self.models_dir, "hype_model.pkl")
        self.trend_model_path = os.path.join(self.models_dir, "trend_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.trend_scaler_path = os.path.join(self.models_dir, "trend_scaler.pkl")
        self.menu_cache_path = os.path.join(self.models_dir, "menu_cache.json")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.menu_cache = self._load_menu_cache()
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.init_csv()
        
    def init_csv(self):
        """Initialize CSV file for storing user selections"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'restaurant_name', 'latitude', 'longitude', 'user_query', 'rating', 'review_count', 'categories', 'price'])

    def load_restaurants(self, location: str = 'Los Angeles', max_results: int = 100):
        """Load restaurant data with progress tracking"""
        try:
            # Load restaurants from Yelp API or local JSON
            self.restaurants_data = self.get_restaurants({'location': location}, max_results)
            print(f"Fetched {len(self.restaurants_data)} restaurants for location '{location}'")
            
            # Try to load existing models first
            if self._load_existing_models():
                print("🚀 Using existing trained models - startup complete!")
                return
            
            print("🔄 No existing models found, creating new ones...")
            self._create_vectorstore()
            self._train_ml_models()
            self._save_models()
            print("✅ All models trained and saved successfully")
            
        except Exception as e:
            print(f"❌ Error loading restaurants: {e}")
    
    def get_restaurants(self, query: dict = None, max_results: int = 100):
        """Fetch restaurants from Yelp API or local JSON if API fails."""
        # First try Yelp API
        try:
            url = 'https://api.yelp.com/v3/businesses/search'
            headers = {'Authorization': f'Bearer {YELP_API_KEY}'}
            
            params = {
                'term': 'restaurant',
                'location': 'Los Angeles',
                'limit': 50,
            }
            if query:
                # Ensure location is properly formatted
                if 'location' in query:
                    params['location'] = query['location'].replace(' ', '+')
                
                # Add other query parameters
                for key, value in query.items():
                    if key != 'location':  # We already handled location
                        params[key] = value
            
            all_businesses = []
            total_to_fetch = min(max_results, 240)  # Yelp API limit
            
            for offset in range(0, total_to_fetch, 50):
                params['offset'] = offset
                params['limit'] = min(50, total_to_fetch - offset)
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                businesses = data.get('businesses', [])
                
                if not businesses:
                    break
                    
                all_businesses.extend(businesses)
                
                if len(businesses) < params['limit']:
                    break
                    
            print(f"✅ Fetched {len(all_businesses)} restaurants from Yelp API")
            return all_businesses
            
        except Exception as e:
            print(f"❌ Yelp API failed: {e}")
            print("🔄 Attempting to load from local dataset...")
            
            # Fallback to local JSON dataset
            json_path = "dataset/yelp_restaurants_full.json"
            try:
                # Fix: Use UTF-8 encoding to handle special characters
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(data, dict) and 'businesses' in data:
                    businesses = data['businesses']
                elif isinstance(data, list):
                    businesses = data
                else:
                    raise ValueError("Unexpected JSON structure")
                    
                # Apply max_results limit
                limited_businesses = businesses[:max_results]
                print(f"✅ Loaded {len(limited_businesses)} restaurants from local dataset")
                return limited_businesses
                
            except Exception as fallback_error:
                print(f"❌ Failed to load local dataset: {fallback_error}")
                print("⚠️  Returning empty restaurant list")
                return []
        
    def _load_menu_cache(self):
        """Load cached menu items"""
        if os.path.exists(self.menu_cache_path):
            try:
                with open(self.menu_cache_path, 'r') as f:
                    print("✅ Menu cache loaded successfully")
                    return json.load(f)
            except:
                print("⚠️  Menu cache corrupted, creating new one")
                return {}
        return {}
    
    def _save_menu_cache(self):
        """Save menu items cache"""
        try:
            with open(self.menu_cache_path, 'w') as f:
                json.dump(self.menu_cache, f)
            print("✅ Menu cache saved successfully")
        except Exception as e:
            print(f"❌ Error saving menu cache: {e}")
    
    def _load_existing_models(self):
        """Load existing trained models if available"""
        try:
            # Check if all model files exist
            model_files = [
                self.hype_model_path,
                self.trend_model_path, 
                self.scaler_path,
                self.trend_scaler_path
            ]
            
            if not all(os.path.exists(f) for f in model_files):
                return False
            
            print("📦 Loading existing ML models...")
            
            # Load ML models
            with open(self.hype_model_path, 'rb') as f:
                self.hype_model = pickle.load(f)
            
            with open(self.trend_model_path, 'rb') as f:
                self.trend_model = pickle.load(f)
                
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(self.trend_scaler_path, 'rb') as f:
                self.trend_scaler = pickle.load(f)
            
            print("✅ ML models loaded successfully")
            
            # Load vectorstore
            if os.path.exists(self.vectorstore_path):
                print("📦 Loading existing vectorstore...")
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vectorstore loaded successfully")
            else:
                print("🔄 Creating new vectorstore...")
                self._create_vectorstore()
                
            return True
            
        except Exception as e:
            print(f"⚠️  Error loading existing models: {e}")
            return False
    
    def _save_models(self):
        """Save trained models for future use"""
        try:
            print("💾 Saving trained models...")
            
            # Save ML models
            with open(self.hype_model_path, 'wb') as f:
                pickle.dump(self.hype_model, f)
                
            with open(self.trend_model_path, 'wb') as f:
                pickle.dump(self.trend_model, f)
                
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            with open(self.trend_scaler_path, 'wb') as f:
                pickle.dump(self.trend_scaler, f)
            
            # Save vectorstore
            if self.vectorstore:
                self.vectorstore.save_local(self.vectorstore_path)
            
            # Save menu cache
            self._save_menu_cache()
            
            print("✅ All models saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def _create_vectorstore(self):
        """Proper vector store creation with embeddings"""
        print("🔄 Creating vectorstore...")
        documents = []
        
        total_restaurants = len(self.restaurants_data)
        for idx, restaurant in enumerate(self.restaurants_data):
            if idx % 100 == 0:
                print(f"   Processing restaurant {idx+1}/{total_restaurants}...")
                
            categories = ", ".join([cat['title'] for cat in restaurant.get('categories', [])])
            
            # Use cached menu items or generate new ones
            restaurant_id = restaurant.get('id', f"restaurant_{idx}")
            if restaurant_id in self.menu_cache:
                menu_items = self.menu_cache[restaurant_id]
            else:
                menu_items = self._extract_menu_items(restaurant)
                self.menu_cache[restaurant_id] = menu_items
            
            text = f"""
            Name: {restaurant.get('name', '')}
            Categories: {categories}
            Rating: {restaurant.get('rating', 0)}
            Price: {restaurant.get('price', 'N/A')}
            Location: {restaurant.get('location', {}).get('display_address', [])}
            Review Count: {restaurant.get('review_count', 0)}
            Phone: {restaurant.get('display_phone', '')}
            Popular Menu Items: {menu_items}
            Transactions: {', '.join(restaurant.get('transactions', []))}
            Distance: {restaurant.get('distance', 'N/A')} meters
            """
            
            doc = Document(
                page_content=text,
                metadata={
                    'id': restaurant_id,
                    'name': restaurant.get('name'),
                    'rating': restaurant.get('rating', 0),
                    'review_count': restaurant.get('review_count', 0),
                    'categories': categories,
                    'price': restaurant.get('price', 'N/A'),
                    'coordinates': restaurant.get('coordinates', {}),
                    'location': restaurant.get('location', {}),
                    'menu_items': menu_items,
                    'transactions': restaurant.get('transactions', [])
                }
            )
            documents.append(doc)
        
        # Create vector store with optimized parameters
        print("🔄 Generating embeddings for document chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"🔄 Generating embeddings for {len(splits)} chunks...")
        
        # Create vector store directly with embeddings
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        print("✅ Vector store created successfully")
        return self.vectorstore
    
    def _extract_menu_items(self, restaurant: Dict) -> str:
        """Extract/generate popular menu items using LLM"""
        categories = [cat['title'] for cat in restaurant.get('categories', [])]
        restaurant_name = restaurant.get('name', '')
        
        prompt = f"""Based on the restaurant name "{restaurant_name}" and categories "{', '.join(categories)}", 
        suggest 3-5 popular menu items that would typically be served at this type of restaurant.
        
        Return only the menu items separated by commas, no explanations.
        Example: "Margherita Pizza, Caesar Salad, Tiramisu, Carbonara Pasta"
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"❌ Error generating menu items: {e}")
            # Fallback to a generic response
            return "Popular dishes available"
        
    def _calculate_operational_intensity(self, business_hours):
        """Precise operational intensity calculation"""
        if not business_hours:
            return 0.0
        
        total_hours = 0.0
        open_days = set()
        weekend_hours = 0
        meal_types = set()
        
        for schedule in business_hours:
            for period in schedule.get('open', []):
                start = int(period['start'][:2]) + int(period['start'][2:])/60.0
                end = int(period['end'][:2]) + int(period['end'][2:])/60.0
                duration = end - start
                total_hours += duration
                open_days.add(period['day'])
                
                # Weekend hours (Friday=5, Saturday=6)
                if period['day'] in [5, 6]:
                    weekend_hours += duration
                
                # Meal type detection
                if start <= 11: meal_types.add('breakfast')
                if 11 < start <= 15: meal_types.add('lunch')
                if start > 15: meal_types.add('dinner')
        
        # Calculate metrics
        days_open = len(open_days)
        meal_variety = len(meal_types) / 3.0
        
        return (
            0.3 * days_open / 7.0 +
            0.3 * min(total_hours / 84.0, 1.0) +  # Max 84 hours/week
            0.2 * min(weekend_hours / 32.0, 1.0) +  # Max 16 hours/day weekend
            0.2 * meal_variety
        )
    
    PREMIUM_ZIPS = ['90036', '90026', '90027', '90293', '90402', '90019', '90022', '91401', '90066', '90077', '90046', '91501', '90049', '91020']

    def _extract_features_for_ml(self, restaurant: Dict) -> List[float]:
        """Highly accurate feature extraction for scoring"""
        
        # Basic features
        rating = restaurant.get('rating', 0.0)
        review_count = restaurant.get('review_count', 0)
        
        # Normalized popularity metrics
        rating_factor = min(rating * 20, 100)  # Convert to 0-100 scale
        review_factor = min(np.log1p(review_count) * 15, 100)  # Logarithmic scaling
        
        # Price features
        price_str = restaurant.get('price', '$')
        price_level = len(price_str) if isinstance(price_str, str) else 1
        price_factor = (4 - min(price_level, 4)) * 15  # Higher price = better
        
        # Category features
        categories = restaurant.get('categories', [])

        # Location features
        location = restaurant.get('location', {})
        zip_code = location.get('zip_code', '00000')
        premium_zip = 1.0 if zip_code in self.PREMIUM_ZIPS else 0.0
        
        # Operational features
        op_intensity = self._calculate_operational_intensity(restaurant.get('business_hours', []))
        
        # Service features
        transactions = restaurant.get('transactions', [])
        has_delivery = 1.0 if 'delivery' in transactions else 0.0
        has_reservation = 1.0 if 'reservation' in transactions else 0.0
        
        # Digital presence
        attributes = restaurant.get('attributes', {})
        has_menu = 1.0 if attributes.get('menu_url') else 0.0
        has_image = 1.0 if restaurant.get('image_url') else 0.0
        
        return [
            rating_factor,
            review_factor,
            price_factor,
            op_intensity * 100,
            premium_zip * 20,
            has_delivery * 10,
            has_reservation * 10,
            has_menu * 10,
            has_image * 10,
            len(categories) * 5
        ]

    def _train_ml_models(self):
        """Train high-precision models with accurate targets"""
        if not self.restaurants_data:
            print("❌ No restaurant data available for training")
            return
            
        print("🤖 Training high-precision ML models...")
        features = []
        hype_targets = []
        trend_targets = []
        
        total_restaurants = len(self.restaurants_data)
        for idx, restaurant in enumerate(self.restaurants_data):
            if idx % 200 == 0:
                print(f"   Processing restaurant {idx+1}/{total_restaurants}...")
                
            # Extract features
            feature_vector = self._extract_features_for_ml(restaurant)
            features.append(feature_vector)
            
            # Extract key properties
            rating = restaurant.get('rating', 0.0)
            review_count = restaurant.get('review_count', 0)
            categories = [cat['alias'] for cat in restaurant.get('categories', [])]
            zip_code = restaurant.get('location', {}).get('zip_code', '')
            
            # Calculate hype score
            hype = (
                0.45 * min(rating * 20, 100) +          # Rating component
                0.35 * min(np.log1p(review_count) * 15, 100) +  # Review component
                0.10 * (4 - min(len(restaurant.get('price', '$')), 4)) * 15 +
                0.20 * self._calculate_operational_intensity(restaurant.get('business_hours', [])) * 100 +
                0.15 * (1 if zip_code in self.PREMIUM_ZIPS else 0) * 25 +
                0.10 * (1 if 'delivery' in restaurant.get('transactions', []) else 0) * 15 +
                0.05 * (1 if 'reservation' in restaurant.get('transactions', []) else 0) * 10 +
                0.05 * (1 if restaurant.get('image_url') else 0) * 15
            )
            hype = min(100, max(0, hype))
            hype_targets.append(hype)
            
            # Calculate trend score
            trend = (
                0.50 * hype +
                0.35 * max((rating - 4.0) * 25, 0) +  # Bonus for high ratings
                0.30 * min(np.log1p(review_count) * 10, 50) +  # Review momentum
                0.10 * (1 if restaurant.get('attributes', {}).get('menu_url') else 0) * 20 +
                0.15 * (1 if zip_code in self.PREMIUM_ZIPS else 0) * 15 +
                0.25 * self._calculate_operational_intensity(restaurant.get('business_hours', [])) * 100 +
                0.15 * (1 if 'delivery' in restaurant.get('transactions', []) else 0) * 15 +
                0.10 * (1 if 'reservation' in restaurant.get('transactions', []) else 0) * 10
            )
            trend = min(100, max(0, trend))
            trend_targets.append(trend)
        
        # Train hype model
        X = np.array(features)
        y_hype = np.array(hype_targets)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.hype_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        self.hype_model.fit(X_scaled, y_hype)
        
        # Train trend model
        self.trend_scaler.fit(X)
        X_trend_scaled = self.trend_scaler.transform(X)
        
        self.trend_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            max_features=0.8,
            random_state=42
        )
        self.trend_model.fit(X_trend_scaled, trend_targets)
        
        # Model diagnostics
        hype_r2 = self.hype_model.score(X_scaled, y_hype)
        trend_r2 = self.trend_model.score(X_trend_scaled, trend_targets)
        print(f"✅ Models trained - Hype R²: {hype_r2:.3f}, Trend R²: {trend_r2:.3f}")
        print(f"   Hype scores: Min={min(y_hype):.1f}, Avg={np.mean(y_hype):.1f}, Max={max(y_hype):.1f}")
        print(f"   Trend scores: Min={min(trend_targets):.1f}, Avg={np.mean(trend_targets):.1f}, Max={max(trend_targets):.1f}")

    def predict_hype_score(self, restaurant: Dict) -> float:
        """Predict current hype score for a restaurant"""
        features = np.array([self._extract_features_for_ml(restaurant)])
        
        features_scaled = self.scaler.transform(features)
        hype_score = self.hype_model.predict(features_scaled)[0]
        return max(0, min(100, hype_score))
    
    def predict_future_popularity(self, restaurant: Dict) -> float:
        """Predict future popularity trend"""
            
        features = np.array([self._extract_features_for_ml(restaurant)])
        features_trend_scaled = self.trend_scaler.transform(features)
        trend_score = self.trend_model.predict(features_trend_scaled)[0]
        return max(0, min(100, trend_score))
    
    def generate_ai_summary(self, restaurant: Dict) -> str:
        """Generate AI-powered restaurant summary"""
        categories = [cat['title'] for cat in restaurant.get('categories', [])]
        menu_items = self._extract_menu_items(restaurant)
        
        prompt = f"""Generate a compelling 2-3 sentence summary for this restaurant:
        
        Name: {restaurant.get('name')}
        Categories: {', '.join(categories)}
        Rating: {restaurant.get('rating')} stars
        Reviews: {restaurant.get('review_count')} reviews
        Popular items: {menu_items}
        
        Make it engaging and highlight what makes this place special. Focus on the food, atmosphere, and why someone should visit."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"A popular {', '.join(categories[:2]).lower()} restaurant with {restaurant.get('rating')} stars and {restaurant.get('review_count')} reviews, known for quality food and good service."
    
    def save_selection(self, restaurant_name: str, coordinates: Dict, user_query: str, 
                      rating: float = 0, review_count: int = 0, categories: str = "", price: str = ""):
        """Save user restaurant selection to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lat = coordinates.get('latitude', '')
        lng = coordinates.get('longitude', '')
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, restaurant_name, lat, lng, user_query, rating, review_count, categories, price])
        
        print(f"Saved selection: {restaurant_name}")
