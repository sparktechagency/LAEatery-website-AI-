import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
from data_manager import RestaurantDataManager, get_day_part
from ai_tools import RestaurantConcierge

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

YELP_API_KEY = os.getenv("YELP_API_KEY")

# Global conversation history
conversation_history = []

# Initialize concierge
concierge = RestaurantConcierge()

# Load restaurant data on startup
if YELP_API_KEY:
    concierge.data_manager.load_restaurants('Los Angeles', 240)
else:
    print("Warning: Restaurant data file not found. Please ensure 'yelp_restaurants_full.json' exists.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Enhanced chat endpoint with conversation flow"""
    try:
        data = request.json
        print(f"Received chat request: {data}")
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process message through enhanced concierge
        response = concierge.chat(user_message)
        
        return jsonify({
            'response': response,
            'conversation_stage': concierge.conversation_stage,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint1():
    """Enhanced chat endpoint with conversation flow and user_id param"""
    try:
        user_id = request.args.get('user_id', None)  # Get user_id from URL params
        data = request.json or {}
        user_message = data.get('message', '')
        
        print(f"Received chat request from user_id={user_id}: {data}")
        
        if not user_id:
            return jsonify({'error': 'Missing user_id in query parameters'}), 400
        
        if not user_message:
            return jsonify({'error': 'No message provided in JSON body'}), 400
        
        # You can optionally use user_id for per-user session or logging here
        
        response = concierge.chat(user_message)
        
        return jsonify({
            'user_id': user_id,
            'response': response,
            'conversation_stage': concierge.conversation_stage,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trending', methods=['GET'])
def get_trending_restaurants():
    """Get currently trending restaurants - future predictions only on explicit request"""
    try:
        show_future = request.args.get('future', 'false').lower() == 'true'
       
        # Get top trending restaurants based on current hype scores
        trending_data = []
       
        for restaurant in concierge.data_manager.restaurants_data[:20]: 
            hype_score = concierge.data_manager.predict_hype_score(restaurant)
           
            if hype_score > 70:  # Only high-hype restaurants
                restaurant_data = {
                    'id': restaurant.get('id'),
                    'alaias': restaurant.get('alias'),
                    'name': restaurant.get('name'),
                    'is_closed': restaurant.get('is_closed', False),
                    'category': ", ".join([cat['title'] for cat in restaurant.get('categories', [])]),
                    'price': restaurant.get('price', 'N/A'),
                    'phone': restaurant.get('phone', 'N/A'),
                    'display_phone': restaurant.get('display_phone', 'N/A'),
                    'distance': round(restaurant.get('distance', 0)),
                    'business_hours': restaurant.get('business_hours', []),
                    'url': restaurant.get('url'),
                    'image_url': restaurant.get('image_url'),
                    'rating': restaurant.get('rating'),
                    'hype_score': round(hype_score, 1),
                    'review_count': restaurant.get('review_count'),
                    'transaction_types': ", ".join(restaurant.get('transactions', [])),
                    'address': ", ".join(restaurant.get('location', {}).get('display_address', [])),
                    'city': restaurant.get('location', {}).get('city', ''),
                    'state': restaurant.get('location', {}).get('state', ''),
                    'zip_code': restaurant.get('location', {}).get('zip_code', ''),
                    'country': restaurant.get('location', {}).get('country', ''),
                    'business_hours': restaurant.get('business_hours', []),
                    'attributes': restaurant.get('attributes', {}),
                    'coordinates': restaurant.get('coordinates', {}),
                    'location': restaurant.get('location', {})
                }
               
                # Only add future trend data if explicitly requested
                if show_future:
                    future_score = concierge.data_manager.predict_future_popularity(restaurant)
                    restaurant_data['future_trend'] = round(future_score, 1)
                    restaurant_data['prediction'] = 'Rising Star' if future_score > hype_score + 5 else 'Currently Hot'
               
                trending_data.append(restaurant_data)
       
        # Sort by current hype score (or future trend if requested)
        sort_key = 'future_trend' if show_future else 'hype_score'
        trending_data.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
       
        return jsonify({
            'trending_restaurants': trending_data[:10],
            'showing_future_predictions': show_future,
            'timestamp': datetime.now().isoformat()
        })
   
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/health')
def health_check():
    """Enhanced health check with model status"""
    return jsonify({
        'status': 'healthy',
        'restaurants_loaded': len(concierge.data_manager.restaurants_data),
        'vectorstore_ready': concierge.data_manager.vectorstore is not None,
        'hype_model_ready': concierge.data_manager.hype_model is not None,
        'trend_model_ready': concierge.data_manager.trend_model is not None,
        'yelp_api_available': YELP_API_KEY is not None,
        'conversation_history_length': len(conversation_history),
        'current_time': get_day_part(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get user selection analytics"""
    try:
        if not os.path.exists(concierge.data_manager.csv_file):
            return jsonify({'message': 'No analytics data available'})
        
        df = pd.read_csv(concierge.data_manager.csv_file)
        
        analytics = {
            'total_selections': len(df),
            'popular_cuisines': df['categories'].value_counts().head(5).to_dict() if 'categories' in df.columns else {},
            'average_rating': df['rating'].mean() if 'rating' in df.columns else 0,
            'price_distribution': df['price'].value_counts().to_dict() if 'price' in df.columns else {},
            'recent_selections': df.tail(10).to_dict('records') if len(df) > 0 else [],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(analytics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    global conversation_history
    conversation_history = []
    concierge.conversation_stage = "greeting"
    concierge.memory.clear()
    
    return jsonify({
        'message': 'Conversation reset successfully',
        'timestamp': datetime.now().isoformat()
    })

# if __name__ == "__main__":
#     print("🍽️ Starting Enhanced Restaurant AI Concierge...")
#     print(f"Current time period: {get_day_part()}")
#     print(f"Yelp API available: {'Yes' if YELP_API_KEY else 'No'}")
#     app.run(debug=True, host='0.0.0.0', port=5000)
