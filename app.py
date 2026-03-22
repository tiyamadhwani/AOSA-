"""
aosa Bakehouse & Roastery — Order & Analytics Platform
Flask + SQLite + Google Gemini
"""

import os, json, uuid, sqlite3, random, requests as http_requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google import genai

# ── LANGCHAIN IMPORTS (pip install langchain langchain-google-genai langchain-core) ──
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool as lc_tool
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False
    print("⚠️  LangChain not installed — using legacy Gemini chat.\n"
          "   Run:  pip install langchain langchain-google-genai langchain-core")

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyDQSdiQtiMl3pQGlPTLgAdDO1FbqvL7EPA')
gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

_HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=_HERE, static_url_path='')
DB_PATH = os.path.join(_HERE, 'aosa.db')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# ── PAYPAL CONFIG ──────────────────────────────
PAYPAL_CLIENT_ID = os.environ.get('PAYPAL_CLIENT_ID', 'YOUR_PAYPAL_CLIENT_ID_HERE')
PAYPAL_CLIENT_SECRET = os.environ.get('PAYPAL_CLIENT_SECRET', 'YOUR_PAYPAL_SECRET_HERE')
PAYPAL_BASE = os.environ.get('PAYPAL_BASE', 'https://api-m.sandbox.paypal.com')

def get_paypal_token():
    try:
        r = http_requests.post(
            f'{PAYPAL_BASE}/v1/oauth2/token',
            headers={'Accept': 'application/json', 'Accept-Language': 'en_US'},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
            data={'grant_type': 'client_credentials'},
            timeout=10
        )
        return r.json().get('access_token')
    except Exception as e:
        return None

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,X-Admin-Token'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return r

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ── SCHEMA ────────────────────────────────────
def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS venues (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
            description TEXT, address TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY, venue_id TEXT NOT NULL REFERENCES venues(id) ON DELETE CASCADE,
            name TEXT NOT NULL, sort_order INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS menu_items (
            id TEXT PRIMARY KEY, venue_id TEXT NOT NULL REFERENCES venues(id) ON DELETE CASCADE,
            category_id TEXT REFERENCES categories(id) ON DELETE SET NULL,
            name TEXT NOT NULL, description TEXT, price REAL NOT NULL,
            is_veg INTEGER DEFAULT 0, is_vegan INTEGER DEFAULT 0,
            is_available INTEGER DEFAULT 1, tags TEXT DEFAULT '[]', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, venue_id TEXT NOT NULL, customer_name TEXT, table_ref TEXT,
            order_type TEXT NOT NULL, spice_level TEXT, dietary_pref TEXT DEFAULT '[]',
            portion_size TEXT, special_instructions TEXT, total_amount REAL DEFAULT 0,
            status TEXT DEFAULT 'pending', created_at TEXT, hour_of_day INTEGER, day_of_week TEXT
        );
        CREATE TABLE IF NOT EXISTS order_items (
            id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
            menu_item_id TEXT NOT NULL, name TEXT NOT NULL, price REAL NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY, venue_id TEXT NOT NULL, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS tables (
            id TEXT PRIMARY KEY,
            venue_id TEXT NOT NULL REFERENCES venues(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            sort_order INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_orders_venue ON orders(venue_id);
        CREATE INDEX IF NOT EXISTS idx_orders_hour  ON orders(hour_of_day);
        CREATE INDEX IF NOT EXISTS idx_items_venue  ON menu_items(venue_id);
        CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
    """)
    db.commit()
    if db.execute("SELECT COUNT(*) FROM venues").fetchone()[0] == 0:
        _seed(db)
    db.close()

# ── AOSA FULL MENU SEED ────────────────────────
def _seed(db):
    vid = str(uuid.uuid4())
    db.execute("INSERT INTO venues (id,name,type,description,address,created_at) VALUES (?,?,?,?,?,?)",
               (vid, 'aosa', 'cafe',
                'Bakehouse & Roastery — A curated selection of culinary treasures, crafted with passion & creativity',
                'Local Café', now_str()))

    menus = {
        'All Day Breakfast': [
            ('French Omelette', 'Served with baby potatoes & pan seared cherry tomatoes', 300, 1, 0, ['eggs','breakfast','light']),
            ('Masala Omelette', 'Onion | Tomato | Chilli | Coriander', 300, 1, 0, ['eggs','breakfast','spicy','indian']),
            ('Cheese Omelette', 'English Cheddar', 300, 1, 0, ['eggs','breakfast','cheesy']),
            ('Skinny Omelette', 'Egg Whites — light and healthy', 300, 1, 0, ['eggs','breakfast','healthy','light']),
            ('Eggs Ben-Addict', 'Poached Eggs | Chicken Ham | English Muffin | Hollandaise', 350, 0, 0, ['eggs','breakfast','hearty']),
            ('Eggs Florentine', 'Poached Eggs | Sautéed Spinach | English Muffin | Hollandaise', 350, 1, 0, ['eggs','breakfast','vegetarian','light']),
            ('Mushroom & Bell Pepper Frittata', 'Light pastry topped with creamy mushroom and parmesan', 350, 1, 0, ['eggs','breakfast','vegetarian','mushroom']),
            ('Mustard Upma', 'Quinoa | Tomato & Coconut Chutney | Cashew | Curry Leaf', 350, 1, 1, ['breakfast','indian','vegan','healthy']),
        ],
        'Toasts & Pancakes': [
            ('Shakshuka with Toast', 'Wilted Spinach | Spicy Sauce', 350, 1, 0, ['eggs','spicy','vegetarian','toast']),
            ('Nutella French Toast', 'Whipped Cream | Hazelnuts', 400, 1, 0, ['sweet','toast','indulgent','nutella']),
            ('Pancake Stack', 'Mix Fruit Jam | Whipped Cream | Banana Caramel Sauce', 400, 1, 0, ['sweet','pancakes','breakfast','indulgent']),
            ('Buckwheat & Chickpea Pancakes', 'Hummus | Homemade Chutneys | House Rucola Salad', 350, 1, 1, ['healthy','vegan','pancakes','light']),
        ],
        'Sandwiches': [
            ('Italian Sandwich', 'Fresh Mozzarella | Rocket Leaf | Focaccia', 450, 1, 0, ['sandwich','vegetarian','italian','light']),
            ('Med Pita Sandwich', 'Avocado | Cucumber | Hummus', 400, 1, 1, ['sandwich','vegan','healthy','light']),
            ('Chipotle Chicken Sandwich', 'Fried Egg | Focaccia', 500, 0, 0, ['sandwich','chicken','spicy','hearty']),
            ('Lamb Sandwich', 'Rocket | Feta | Lamb Pepperoni | Pepper Relish | Ciabatta', 525, 0, 0, ['sandwich','lamb','hearty','premium']),
        ],
        'Sides': [
            ('Sides Platter', 'Toast | Baked Beans | Hash Browns | Chicken Sausages | Potato Wedges', 100, 0, 0, ['sides','light','breakfast']),
        ],
        'Smoothie Jars': [
            ('Green Apple & Avocado Smoothie Jar', 'Granola | Chia | Banana | Coconut Flakes', 350, 1, 1, ['healthy','smoothie','vegan','fresh']),
            ('Berry Smoothie Jar', 'Granola | Berries | Yoghurt', 350, 1, 0, ['healthy','smoothie','fresh','sweet']),
            ('Chocolate & Coconut Smoothie Jar', 'Dates | Walnuts | Brownie Bites', 350, 1, 1, ['sweet','smoothie','vegan','indulgent']),
        ],
        'Salads': [
            ('Rocket & Red Wine Poached Pear Salad', 'Mango Chunda | Feta', 520, 1, 0, ['salad','light','vegetarian','healthy']),
            ('Quinoa Chicken Salad', 'Chilli | Sweet Peas', 525, 0, 0, ['salad','healthy','protein','light']),
            ('Barley Moong Sprout Salad', 'Corn | Savoury Schnapps', 400, 1, 1, ['salad','healthy','vegan','light']),
        ],
        'Flatbreads': [
            ('Italian Lifestyle Flatbread', 'Basil | Mozzarella', 500, 1, 0, ['flatbread','vegetarian','italian','cheesy']),
            ('Spinachi Truffle Flatbread', 'Truffle Oil | Spinach | Garlic', 580, 1, 0, ['flatbread','vegetarian','premium','truffle']),
            ('Pesto Balsamic Flatbread', 'Pesto | Pinenut | Rocket Leaves', 650, 1, 0, ['flatbread','vegetarian','premium']),
            ('Pepper Shroom Flatbread', 'Bell Pepper | Mushroom', 550, 1, 1, ['flatbread','vegan','vegetarian']),
            ('Pesto Paneer Flatbread', 'Onion | Pinenut | Rucola', 580, 1, 0, ['flatbread','vegetarian','paneer']),
            ('Smoked Chicken Flatbread', 'Basil Oil', 625, 0, 0, ['flatbread','chicken','smoky']),
            ('Lamb Pepperoni Flatbread', 'Pickled Chilli | Parmesan Shavings', 650, 0, 0, ['flatbread','lamb','spicy','premium']),
        ],
        'A Little Mix': [
            ('Soya Keema', 'Freshly Baked Pav', 375, 1, 1, ['indian','vegan','spicy','hearty']),
            ('Masala Fish Fingers', 'Garlic Chutney | Jeera Tartar Sauce', 450, 0, 0, ['fish','indian','spicy','crispy']),
            ('Aloo Bravas', 'Pepper Refresh Relish | Potato Foam', 400, 1, 1, ['potato','vegan','spicy','crispy']),
            ('Massaman Curry', 'Jasmine Rice', 575, 1, 1, ['curry','vegan','thai','hearty']),
            ('Tamarind Curry', 'Husked Barley', 550, 1, 1, ['curry','vegan','indian','tangy']),
            ('Chifferi Cacio e Pepe', 'Parmesan | Black Pepper | Cream', 475, 1, 0, ['pasta','vegetarian','cheesy','creamy']),
            ('Fettuccini Arrabiata', 'Tomato | Chilli | Basil', 450, 1, 1, ['pasta','vegan','spicy','italian']),
            ('Penne Pesto', 'Pesto | Chicken | Parmesan', 550, 0, 0, ['pasta','chicken','creamy']),
            ('Linguini Aglio e Olio', 'Orange | Prawns | Butter & Parmesan Emulsion', 650, 0, 0, ['pasta','seafood','premium','butter']),
            ('Healing Vegetable Kedgeree', 'Roasted Papadums', 350, 1, 1, ['indian','vegan','healthy','rice']),
        ],
        'Aosa Specials': [
            ('Mezze Platter', 'A selection of house mezze', 599, 1, 0, ['sharing','vegetarian','light']),
            ('Burrito Bowl', 'Hearty burrito bowl with assorted condiments', 499, 1, 0, ['mexican','vegetarian','hearty']),
            ('Nacho Bowl', 'Classic nacho bowl', 499, 1, 0, ['mexican','vegetarian','cheesy','crispy']),
            ('Mushroom Tartine on Toast', 'Add Poached Egg ₹150 | Add Chicken Ham ₹150', 380, 1, 0, ['toast','vegetarian','mushroom','light']),
            ('Homemade Soft Shell Fried Chicken Tacos', 'Crispy fried chicken tacos', 475, 0, 0, ['tacos','chicken','crispy','hearty']),
            ('Homemade Soft Shell Fried Paneer Tacos', 'Crispy fried paneer tacos', 450, 1, 0, ['tacos','vegetarian','paneer','crispy']),
            ('Ham Mustard & Cheese Croissant', 'Buttery croissant with ham and cheese', 400, 0, 0, ['croissant','ham','cheesy']),
            ('Caprese Croissant', 'Fresh caprese in a flaky croissant', 350, 1, 0, ['croissant','vegetarian','fresh']),
            ('Cheese Omelette Croissant', 'Omelette tucked inside a warm croissant', 350, 1, 0, ['croissant','vegetarian','eggs']),
            ('Pistachio Crusted Grilled Chicken', 'With English Vegetables Mash', 499, 0, 0, ['chicken','premium','hearty']),
        ],
        'Hot Coffee': [
            ('Espresso', 'A concentrated shot of coffee', 140, 1, 1, ['coffee','hot','strong']),
            ('Ristretto', 'More concentrated and shorter than espresso', 140, 1, 1, ['coffee','hot','strong']),
            ('Macchiato', 'Espresso with a dash of milk froth', 140, 1, 1, ['coffee','hot']),
            ('Americano', 'Espresso topped with hot water', 180, 1, 1, ['coffee','hot','light']),
            ('Long Black', 'Double espresso with hot water — stronger', 200, 1, 1, ['coffee','hot','strong']),
            ('Cappuccino', 'Coffee, warm milk and lots of milk foam', 240, 1, 0, ['coffee','hot','creamy','signature']),
            ('Cafe Latte', 'Hot coffee with steamed milk and less foam', 240, 1, 0, ['coffee','hot','creamy','mild']),
            ('Flat White', 'Steamed milk, almost no foam — hottest milk option', 240, 1, 0, ['coffee','hot','creamy']),
            ('Cortado', 'Shorter than a latte, more milk, more strength', 240, 1, 0, ['coffee','hot','strong']),
            ('Cafe Latte Flavoured', 'Latte with your choice of flavour', 250, 1, 0, ['coffee','hot','sweet','flavoured']),
            ('Mocha', 'Coffee & chocolate together — a perfect match', 260, 1, 0, ['coffee','hot','chocolate','sweet']),
            ('Salted Caramel Popcorn Latte', 'Cinema salted caramel popcorn but in a coffee cup', 280, 1, 0, ['coffee','hot','sweet','salted caramel','signature']),
            ('Spiced C&C Coffee & Cacao', 'Coffee, chocolate and cinnamon spice', 240, 1, 0, ['coffee','hot','spiced','chocolate']),
        ],
        'Cold Brew': [
            ('Cold Brew Classic', 'Classic cold brew served with ice', 190, 1, 1, ['coffee','cold','signature']),
            ('Flavoured Cold Brew', 'Cold brew with your choice of flavour', 210, 1, 1, ['coffee','cold','flavoured']),
            ('Cold Brew Latte', 'Cold brew and milk together', 210, 1, 0, ['coffee','cold','creamy']),
            ('Cold Brew Latte Flavoured', 'Milky cold brew with your flavour choice', 220, 1, 0, ['coffee','cold','creamy','flavoured']),
            ('Cold Brew Aperol', 'Aperol spritz inspired cold brew drink', 220, 1, 1, ['coffee','cold','unique']),
            ('Coffee Mojito', 'Fresh mojito with cold brew', 240, 1, 1, ['coffee','cold','fresh','mocktail']),
            ('Coffee Tonic', 'Coffee and tonic with lemon slice — best in town', 250, 1, 1, ['coffee','cold','tonic','unique']),
            ('Aosa Coffee Tonic', "aosa's special version of coffee tonic", 260, 1, 1, ['coffee','cold','signature','unique']),
        ],
        'Iced Coffee': [
            ('Iced Latte', 'Simple cold and iced milk coffee — peoples choice', 240, 1, 0, ['coffee','cold','creamy','popular']),
            ('Iced Flavoured Latte', 'Iced latte with your choice of flavour', 260, 1, 0, ['coffee','cold','sweet','flavoured']),
            ('Iced Magic Latte', 'Cold coffee with sweetened milk — magic in a cup', 260, 1, 0, ['coffee','cold','sweet']),
            ('Iced Mocha', 'Coffee and chocolate in an iced version — heavenly', 260, 1, 0, ['coffee','cold','chocolate','sweet']),
            ('A Very Berry Cafe Latte', 'Iced coffee with milk and strawberry', 280, 1, 0, ['coffee','cold','berry','fruity']),
            ('Cafe Frappe', 'Classic blended cold coffee', 250, 1, 0, ['coffee','cold','blended']),
            ('Salted Caramel Popcorn Frappe', 'Cinema salted caramel popcorn as a frappe', 280, 1, 0, ['coffee','cold','sweet','salted caramel']),
            ('Caramelised Banana & Vanilla Frappe', 'Banoffee and coffee and vanilla in cold form', 280, 1, 0, ['coffee','cold','sweet','banana']),
            ('Flavoured Frappe', 'Classic coffee frappe with your flavour choice', 260, 1, 0, ['coffee','cold','flavoured']),
            ('Cheesecake Frappe', 'A cold coffee or a cheesecake? Both.', 280, 1, 0, ['coffee','cold','sweet','cheesecake']),
            ('Espresso on the Rocks', 'Like whisky on rocks — but espresso', 150, 1, 1, ['coffee','cold','strong']),
            ('Iced Americano', 'Cold black coffee, espresso, ice & water', 180, 1, 1, ['coffee','cold','strong','light']),
            ('Iced Long Black', 'Double espresso, cold, ice and water', 200, 1, 1, ['coffee','cold','strong']),
        ],
        'Aosa Coffee Specials': [
            ('Vietnamese Styled Hot Coffee', 'Vietnamese style with condensed milk, hot version', 280, 1, 0, ['coffee','hot','vietnamese','sweet','signature']),
            ('Vietnamese Styled Iced Coffee', 'Vietnamese style with condensed milk, iced shaken — top selling', 280, 1, 0, ['coffee','cold','vietnamese','sweet','signature']),
            ('Espresso Martini', 'Virgin espresso martini — must try', 260, 1, 0, ['coffee','cold','mocktail','premium']),
            ('Espresso Bull', 'Caffeine max — espresso and red bull', 300, 1, 1, ['coffee','cold','energy','strong']),
            ('South Indian Filter Coffee', 'Classic South Indian filter coffee', 240, 1, 0, ['coffee','hot','south indian','traditional']),
            ('Hot Chocolate', 'Rich warming hot chocolate', 240, 1, 0, ['chocolate','hot','sweet','comfort']),
        ],
        'Affogato': [
            ('Coffee Mochagato', 'Chocolate icecream, espresso and pink salt', 150, 1, 0, ['coffee','dessert','chocolate','sweet']),
            ('Coffee Affogato', 'Classic vanilla ice cream and espresso shot', 180, 1, 0, ['coffee','dessert','vanilla','sweet']),
            ('Coffee Conegato', "aosa's twist on affogato — must try", 200, 1, 0, ['coffee','dessert','signature','sweet']),
        ],
        'Tea': [
            ('Masala Chai Pot', 'Classic Indian spiced chai', 180, 1, 0, ['tea','hot','spiced','indian']),
            ('Single Malt Grand Bru Assam Leaves', 'Premium Assam tea', 210, 1, 0, ['tea','hot','premium','assam']),
            ('White Tea Saffron Leaves', 'Delicate white tea with saffron', 210, 1, 1, ['tea','hot','premium','light']),
            ('Jasmine Hot Tea', 'Floral jasmine tea', 210, 1, 1, ['tea','hot','floral','light']),
            ('Macha Latte', 'Warm matcha latte', 240, 1, 0, ['tea','hot','matcha','creamy']),
            ('Chamomile Tea', 'Calming chamomile herbal tea', 200, 1, 1, ['tea','hot','herbal','calm','light']),
            ('Iced Macha Latte', 'Cold matcha latte', 240, 1, 0, ['tea','cold','matcha','creamy']),
            ('Iced Espresso Macha Latte', 'Coffee meets matcha in an iced version', 260, 1, 0, ['tea','coffee','cold','unique']),
            ('Classic Lemon Iced Tea', 'Refreshing lemon iced tea', 220, 1, 1, ['tea','cold','lemon','fresh']),
            ('Lemon Mint Iced Tea', 'Lemon and mint iced tea — must try', 220, 1, 1, ['tea','cold','lemon','mint','fresh']),
            ('Strawberry Iced Tea', 'Fruity strawberry iced tea', 220, 1, 1, ['tea','cold','strawberry','fruity']),
            ('Passion Fruit Iced Tea', 'Tropical passion fruit iced tea', 220, 1, 1, ['tea','cold','tropical','fruity']),
        ],
        'Manual Pour Over': [
            ('V60 Hot / Iced', 'V60 pour over — specialty coffee', 240, 1, 1, ['coffee','specialty','filter','pour over']),
            ('Kalita Hot / Iced', 'Kalita wave pour over', 240, 1, 1, ['coffee','specialty','filter','pour over']),
            ('Origami Hot / Iced', 'Origami dripper pour over', 240, 1, 1, ['coffee','specialty','filter','pour over']),
            ('Clever Dripper Drip', 'Immersion style pour over', 240, 1, 1, ['coffee','specialty','filter']),
            ('Clever Dripper Immersion', 'Full immersion brew', 240, 1, 1, ['coffee','specialty','filter']),
        ],
        'Non Coffee Mocktails': [
            ('Mojito', 'Classic fresh mojito', 220, 1, 1, ['mocktail','cold','fresh','mint']),
            ('Flavoured Mojito', 'Mojito with flavour options', 230, 1, 1, ['mocktail','cold','fresh','flavoured']),
            ('Pina Colada', 'Pineapple, coconut and cream — no alcohol but still yummy', 220, 1, 1, ['mocktail','cold','tropical','sweet']),
        ],
        'Shakes': [
            ('Chocolate Shake', 'Rich chocolate milkshake', 240, 1, 0, ['shake','cold','chocolate','sweet']),
            ('Strawberry Shake', 'Fresh strawberry milkshake', 240, 1, 0, ['shake','cold','strawberry','sweet']),
            ('Strawberry Cheesecake Shake', 'Strawberry cheesecake milkshake', 300, 1, 0, ['shake','cold','cheesecake','sweet','premium']),
        ],
        'Soft Drinks': [
            ('Water Bottle', 'Still mineral water', 100, 1, 1, ['water','cold']),
            ('Ginger Ale', 'Refreshing ginger ale', 120, 1, 1, ['soft drink','cold','ginger']),
            ('Tonic Water', 'Premium tonic water', 120, 1, 1, ['soft drink','cold']),
            ('Red Bull', 'Energy drink', 250, 1, 1, ['energy','cold']),
            ('Cold Press Juices', 'Fresh cold pressed juices', 250, 1, 1, ['juice','cold','healthy','fresh']),
        ],
        'Laminated Pastry': [
            ('Aosa Croissant', 'Classic butter croissant — soft, flaky and golden-brown — signature', 200, 1, 0, ['croissant','pastry','buttery','signature','bakery']),
            ('Chocolate Croissant', 'Buttery croissant filled with dark couverture chocolate crème — best selling', 280, 1, 0, ['croissant','pastry','chocolate','sweet','bakery']),
            ('Almond Croissant', 'Sweet almond filling topped with toasted almonds', 280, 1, 0, ['croissant','pastry','almond','sweet','bakery']),
            ('Chocolate Pistachio Cube', 'Flaky cube croissant with chocolate and pistachios — trending', 350, 1, 0, ['croissant','pastry','chocolate','pistachio','premium','bakery']),
            ('Lemon Vanilla Cube', 'Flaky cube croissant with zesty lemon and vanilla custard — trending', 350, 1, 0, ['croissant','pastry','lemon','vanilla','sweet','bakery']),
            ('Mushroom Cream Cheese', 'Flaky pastry topped with creamy mushroom and parmesan', 200, 1, 0, ['pastry','mushroom','savory','cheesy','bakery']),
            ('Veggies Jalapeno & Cheese', 'Flaky pastry with gourmet stuffing and cream cheese', 200, 1, 0, ['pastry','vegetarian','spicy','cheesy','bakery']),
            ('Korean Bun', 'Classic bun with cream cheese and garlic butter', 210, 1, 0, ['bun','korean','cheesy','sweet','bakery']),
            ('Korean Bun 2.0', 'Spinach, mushroom & corn filling — AOSA favourite', 230, 1, 0, ['bun','korean','vegetarian','premium','bakery']),
            ('Margherita Pizza Danish', 'Golden flaky Danish with tomato, mozzarella & basil', 220, 1, 0, ['danish','vegetarian','pizza','cheesy','bakery']),
            ('Spiced Thai Chicken Puff', 'Thai-spiced chicken in flaky pastry', 240, 0, 0, ['puff','chicken','spicy','thai','bakery']),
            ('Bhuna Gosht Puff', 'Tender spiced lamb in buttery pastry', 240, 0, 0, ['puff','lamb','spicy','hearty','bakery']),
        ],
        'Tarts': [
            ('Chocolate Mascarpone Tart', 'Buttery pastry with milk couverture coffee ganache', 250, 1, 0, ['tart','chocolate','sweet','premium','bakery']),
            ('Queen of Tarts', 'Fresh blueberry and mixed berry compote in buttery shortcrust', 300, 1, 0, ['tart','berry','sweet','premium','bakery']),
        ],
        'Cookies': [
            ('Chocolate Hazelnut Cookie', 'Rich chocolate cookie with crunchy hazelnuts', 150, 1, 0, ['cookie','chocolate','hazelnut','sweet','bakery']),
            ('PB & J Cookie', 'Peanut butter and fruity jam in a crunchy cookie', 140, 1, 0, ['cookie','peanut butter','sweet','bakery']),
            ('Aosa OMG Cookie', 'Pistachio paste and berry confit cookie', 150, 1, 0, ['cookie','pistachio','sweet','signature','bakery']),
        ],
        'Entremets & Bistro Style': [
            ('Dessert Island Brownie', 'Brownie with crunchy chocolate coating', 300, 1, 0, ['brownie','chocolate','sweet','indulgent','bakery']),
            ('Biscoff Cheesecake', 'Dense and creamy cheesecake with Biscoff flavour', 320, 1, 0, ['cheesecake','biscoff','sweet','premium','bakery']),
            ('New York Cheesecake', 'American-style authentic creamy cheesecake — best in town', 300, 1, 0, ['cheesecake','sweet','premium','classic','bakery']),
            ('Messy Mud Tub', 'Chocolate paradise layered to perfection in a box', 300, 1, 0, ['chocolate','sweet','indulgent','premium','bakery']),
            ('Chef Special Petit Antoine', 'Hazelnut and French biscuit with dark couverture ganache and sponge', 350, 1, 0, ['dessert','chocolate','premium','signature','bakery']),
            ('Mille-Feuille', 'Laminated puff pastry with rich chocolate cream', 300, 1, 0, ['pastry','chocolate','sweet','french','bakery']),
            ('Mango & Passion Fruit', 'Tropical blend of mango and passion fruit — trending', 280, 1, 1, ['dessert','tropical','sweet','fruity','vegan','bakery']),
            ('Vegan Chocolate Cake', 'Indulgent vegan chocolate cake', 200, 1, 1, ['cake','vegan','chocolate','sweet','bakery']),
            ('Carrot Cake', 'Cinnamon-spiced with cream cheese frosting and walnuts', 190, 1, 0, ['cake','carrot','sweet','classic','bakery']),
            ('Dream Come Blue', 'Blueberry sponge cake — dreamy dessert', 320, 1, 0, ['cake','blueberry','sweet','premium','bakery']),
            ('Tiramisu Tub', 'Classic tiramisu with mascarpone and Kahlua-soaked ladyfingers', 350, 1, 0, ['tiramisu','coffee','sweet','italian','premium','bakery']),
            ('Osaka Style Roll', 'Japanese-style roll with mixed berry compote — must try', 190, 1, 0, ['cake','japanese','berry','sweet','light','bakery']),
        ],
        'Tea Cakes': [
            ('Lamington Bar', 'Layers of lamington, chocolate and coconut — must try', 200, 1, 0, ['cake','chocolate','sweet','coconut','bakery']),
            ('Chocolate and Orange Cake', 'Moist cake with orange flavour (slice ₹200 / loaf ₹600)', 200, 1, 0, ['cake','chocolate','orange','sweet','bakery']),
            ('Lemon Drizzle', 'Lemon teacake with citrus glaze (slice ₹180 / loaf ₹550)', 180, 1, 0, ['cake','lemon','sweet','light','bakery']),
            ('Banana Bread with Walnuts', 'Classic banana bread AOSA twist (slice ₹160 / loaf ₹500)', 160, 1, 0, ['bread','banana','sweet','nutty','bakery']),
            ('Espresso Crumble Cake', 'Espresso-infused crumble cake (slice ₹200 / loaf ₹600)', 200, 1, 0, ['cake','coffee','sweet','premium','bakery']),
        ],
        'Celebration Cakes': [
            ('Lemon Blueberry Cake', '500g ₹1000 / 1000g ₹1800', 1000, 1, 0, ['cake','celebration','lemon','premium','bakery']),
            ('Biscoff Cheesecake Whole', '500g ₹1200 / 1000g ₹2000', 1200, 1, 0, ['cake','celebration','biscoff','premium','bakery']),
            ('Aosa Signature Antoine Cake', '500g ₹1200 / 1000g ₹2000', 1200, 1, 0, ['cake','celebration','signature','premium','bakery']),
            ('Fruit Crumble Cake', '500g ₹1000 / 1000g ₹1800', 1000, 1, 0, ['cake','celebration','fruity','premium','bakery']),
            ('Hazelnut Praline Cake', '500g ₹1200 / 1000g ₹2000', 1200, 1, 0, ['cake','celebration','hazelnut','premium','bakery']),
            ('Espresso Almond Cake', '500g ₹1000 / 1000g ₹1800', 1000, 1, 0, ['cake','celebration','coffee','premium','bakery']),
            ('New York Cheese Cake Whole', '500g ₹1000 / 1000g ₹1800', 1000, 1, 0, ['cake','celebration','cheesecake','premium','bakery']),
            ('100% Chocolate Cake', '500g ₹1200 / 1000g ₹2000', 1200, 1, 0, ['cake','celebration','chocolate','premium','bakery']),
            ('Dream Come Blue Whole', '500g ₹1200 / 1000g ₹2000', 1200, 1, 0, ['cake','celebration','blueberry','premium','bakery']),
        ],
        'Bread': [
            ('Signature Aosa Sourdough', 'Our house sourdough — the one that started it all', 200, 1, 1, ['bread','sourdough','signature','bakery']),
            ('50% Whole Wheat Sourdough', 'Wholesome and nutty whole wheat sourdough', 220, 1, 1, ['bread','sourdough','healthy','whole wheat','bakery']),
            ('Roasted Garlic & Olive Sourdough', 'Garlic and olive oil infused sourdough', 220, 1, 1, ['bread','sourdough','garlic','savory','bakery']),
            ('Pesto & Parmesan Babka', 'Twisted babka with pesto and parmesan', 250, 1, 0, ['bread','babka','pesto','cheesy','bakery']),
            ('Chocolate & Nuts Babka', 'Sweet chocolate babka with mixed nuts', 250, 1, 0, ['bread','babka','chocolate','sweet','bakery']),
            ('Ragi Bread', 'Healthy ragi grain bread', 150, 1, 1, ['bread','ragi','healthy','vegan','bakery']),
            ('Sourdough Focaccia', 'Classic Italian herb focaccia', 200, 1, 1, ['bread','focaccia','italian','vegan','bakery']),
            ('Multigrain Country Loaf', 'Hearty multigrain country loaf', 150, 1, 1, ['bread','multigrain','healthy','vegan','bakery']),
        ],
    }

    for i, (cat_name, items) in enumerate(menus.items()):
        cat_id = str(uuid.uuid4())
        db.execute("INSERT INTO categories (id,venue_id,name,sort_order) VALUES (?,?,?,?)",
                   (cat_id, vid, cat_name, i))
        for item in items:
            name, desc, price, is_veg, is_vegan, tags = item
            db.execute("""INSERT INTO menu_items
                (id,venue_id,category_id,name,description,price,is_veg,is_vegan,tags,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), vid, cat_id, name, desc, price,
                 is_veg, is_vegan, json.dumps(tags), now_str()))

    # Seed sample orders for analytics
    statuses = ['completed','completed','completed','completed','preparing','ready']
    order_types = ['dine-in','dine-in','takeaway']
    spice_levels = ['mild','medium','hot','extra-hot']
    portions = ['small','regular','regular','large']
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    all_items = db.execute("SELECT id,name,price FROM menu_items WHERE venue_id=?", (vid,)).fetchall()
    for _ in range(80):
        hour = random.choices(range(8,23), weights=[1,2,4,6,8,5,3,8,9,10,7,5,8,9,6], k=1)[0]
        dt = datetime.now() - timedelta(days=random.randint(0,6), hours=random.randint(0,3))
        dt = dt.replace(hour=hour)
        oid = str(uuid.uuid4())
        db.execute("""INSERT INTO orders
            (id,venue_id,customer_name,order_type,spice_level,dietary_pref,portion_size,
             total_amount,status,created_at,hour_of_day,day_of_week) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (oid, vid, f"Guest {random.randint(1,99)}", random.choice(order_types),
             random.choice(spice_levels),
             json.dumps(random.sample(['vegetarian','vegan','gluten-free'], k=random.randint(0,1))),
             random.choice(portions), 0, random.choice(statuses),
             dt.strftime('%Y-%m-%d %H:%M:%S'), hour, days[dt.weekday()]))
        chosen = random.sample(list(all_items), min(random.randint(1,3), len(all_items)))
        total = 0
        for item in chosen:
            qty = random.randint(1,2)
            price = float(item['price'])
            db.execute("INSERT INTO order_items (id,order_id,menu_item_id,name,price,quantity) VALUES (?,?,?,?,?,?)",
                       (str(uuid.uuid4()), oid, item['id'], item['name'], price, qty))
            total += price * qty
        db.execute("UPDATE orders SET total_amount=? WHERE id=?", (round(total,2), oid))
    db.commit()
    print("[SEED] aosa Bakehouse & Roastery menu seeded — all 3 menus loaded")

# ── NLP ───────────────────────────────────────
def find_dishes(query, venue_id, top_k=5):
    db = get_db()
    rows = db.execute(
        "SELECT id,name,description,price,is_veg,is_vegan,tags FROM menu_items WHERE venue_id=? AND is_available=1",
        (venue_id,)
    ).fetchall()
    if not rows: return []
    corpus = [f"{r['name']} {r['description']} {' '.join(json.loads(r['tags']))}" for r in rows]
    try:
        mat = TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit_transform(corpus + [query])
        scores = cosine_similarity(mat[-1], mat[:-1])[0]
    except:
        scores = np.zeros(len(rows))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{'id': rows[i]['id'], 'name': rows[i]['name'], 'description': rows[i]['description'],
             'price': rows[i]['price'], 'is_veg': bool(rows[i]['is_veg']),
             'is_vegan': bool(rows[i]['is_vegan']), 'tags': json.loads(rows[i]['tags']),
             'score': round(float(s), 3)} for i, s in ranked if s > 0.01]

# ── LANGCHAIN AGENT CHAT ──────────────────────
def get_ai_chat(messages_history, customer_name, menu_ctx, venue_id=None):
    """
    Mia — LangChain tool-calling agent with full conversation memory.

    The critical difference from the old approach:
    - Old: history was flattened into one big string → model lost track of *what it said*
    - New: history is passed as real HumanMessage / AIMessage objects so the model
      can resolve references like "which one", "the chocolate one", "the first option"
      by looking back at its own prior replies.
    """
    try:
        if not GOOGLE_API_KEY:
            raise Exception("No GOOGLE_API_KEY set")

        name_part = (f" {customer_name}"
                     if customer_name and customer_name.lower() not in ('guest', '')
                     else "")

        # ── System personality ──────────────────
        SYSTEM = f"""You are Mia, the warm, knowledgeable café assistant at aosa Bakehouse & Roastery \
— a beloved local café famous for specialty coffee, artisan croissants, and creative food.

YOUR PERSONALITY
• Warm, genuine, opinionated — like a foodie friend who has tasted everything
• Specific — mention WHY you like something, not just what it is
• Conversational — natural language, contractions, occasional warmth
• Never robotic; avoid bullet lists when a flowing sentence works better

CUSTOMER NAME: {name_part.strip() if name_part else "not provided yet"}

KEY MENU KNOWLEDGE
• Signatures: Aosa Croissant, Cappuccino, Vietnamese Styled Iced Coffee, Cold Brew Classic,
  Chocolate Croissant, New York Cheesecake
• Must-try: Coffee Tonic, Salted Caramel Popcorn Latte, Korean Bun 2.0, Tiramisu Tub
• Prices are in Indian Rupees (₹)

FULL MENU (use search_menu tool for precise lookups):
{menu_ctx}

CONVERSATION MEMORY — CRITICAL RULES
1. You have COMPLETE memory of everything said in this conversation.
2. When the customer uses ANY vague reference — "which one", "that one", "the first",
   "the chocolate one", "option 2", "the second dish you mentioned", etc. — you MUST
   scroll back through your previous replies, find exactly what you listed/described,
   and answer about that specific item. Never claim you don't remember.
3. If you listed 3 desserts and the customer asks "which one has chocolate?" →
   re-read your previous message, identify the chocolate option(s), and reply directly.
4. Always address the customer by name{name_part} when you know it.
5. End with a soft question or offer to help further."""

        # ── Build proper typed message history ──
        history: list = []
        for m in messages_history[:-1]:   # everything except the current query
            if m['role'] == 'user':
                history.append(HumanMessage(content=m['content']))
            else:
                history.append(AIMessage(content=m['content']))

        current_query = messages_history[-1]['content']

        # ── LangChain path ───────────────────────
        if LANGCHAIN_OK:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash',
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=False,
            )

            tools = []
            if venue_id:
                # Capture venue_id in closure so the tool can use it
                _vid = venue_id

                @lc_tool
                def search_menu(query: str) -> str:
                    """Search the aosa menu for dishes by name, ingredient, or tag.
                    Use this whenever you need to find a specific dish or verify details.
                    Input: a natural-language search query."""
                    results = find_dishes(query, _vid, top_k=6)
                    if not results:
                        return "No matching items found for that query."
                    lines = []
                    for d in results:
                        diet = " [Vegan]" if d['is_vegan'] else (" [Veg]" if d['is_veg'] else "")
                        desc = f" — {d['description']}" if d.get('description') else ""
                        lines.append(f"• {d['name']} (₹{d['price']:.0f}){diet}{desc}")
                    return "\n".join(lines)

                tools = [search_menu]

            # Agent with tool-calling (uses scratchpad for tool calls)
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent    = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(
                agent=agent, tools=tools,
                verbose=False, max_iterations=4,
                handle_parsing_errors=True,
            )

            result = executor.invoke({
                "input":        current_query,
                "chat_history": history,
            })
            return result["output"].strip()

        # ── Fallback: raw Gemini with typed messages ─
        # Even without LangChain, pass history as typed objects for better memory
        if not gemini_client:
            raise Exception("gemini_client not available")

        # Build a single structured prompt that preserves roles explicitly
        conv_lines = []
        for m in messages_history[:-1]:
            role = "Customer" if m['role'] == 'user' else "Mia"
            conv_lines.append(f"{role}: {m['content']}")
        conv_block = "\n".join(conv_lines) if conv_lines else "(start of conversation)"

        full_prompt = (
            f"{SYSTEM}\n\n"
            f"=== CONVERSATION HISTORY ===\n{conv_block}\n"
            f"=== END HISTORY ===\n\n"
            f"Customer: {current_query}\nMia:"
        )
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=full_prompt,
        )
        return response.text.strip()

    except Exception as e:
        fallbacks = [
            "Oh, great question! Let me help you find the perfect thing.",
            "Happy to help! What sounds good to you today?",
            "Of course! Tell me what you're in the mood for and I'll point you in the right direction.",
        ]
        return random.choice(fallbacks)

# ── ADMIN AUTH ─────────────────────────────────
def require_admin():
    token = request.headers.get('X-Admin-Token') or request.args.get('token')
    if token != ADMIN_PASSWORD:
        return jsonify({'error': 'Unauthorized'}), 401
    return None

# ═══════════════════════════════════════════════
# PUBLIC ROUTES
# ═══════════════════════════════════════════════

@app.route('/api/venues', methods=['GET'])
def list_venues():
    rows = get_db().execute("SELECT id,name,type,description,address FROM venues ORDER BY name").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/venues/<vid>/menu', methods=['GET'])
def get_menu(vid):
    db = get_db()
    cats = db.execute("SELECT id,name FROM categories WHERE venue_id=? ORDER BY sort_order", (vid,)).fetchall()
    result = []
    for cat in cats:
        items = db.execute(
            "SELECT id,name,description,price,is_veg,is_vegan,tags FROM menu_items WHERE category_id=? AND is_available=1",
            (cat['id'],)
        ).fetchall()
        result.append({
            'category': cat['name'],
            'items': [dict(i)|{'tags':json.loads(i['tags']),'is_veg':bool(i['is_veg']),'is_vegan':bool(i['is_vegan'])} for i in items]
        })
    return jsonify(result)

@app.route('/api/venues/<vid>/chat', methods=['POST'])
def chat(vid):
    data = request.json or {}
    query = data.get('message','').strip()
    session_id = data.get('session_id', str(uuid.uuid4()))
    customer_name = data.get('customer_name','')
    if not query:
        return jsonify({'error': 'message required'}), 400

    db = get_db()
    venue = db.execute("SELECT name FROM venues WHERE id=?", (vid,)).fetchone()
    if not venue: return jsonify({'error': 'Venue not found'}), 404

    # Full menu context for Mia
    items = db.execute(
        "SELECT name,price,is_veg,is_vegan,tags,description "
        "FROM menu_items WHERE venue_id=? AND is_available=1 LIMIT 100",
        (vid,)
    ).fetchall()
    menu_ctx = "\n".join([
        f"- {i['name']} (₹{i['price']:.0f})"
        f"{' [Veg]' if i['is_veg'] else ''}"
        f"{' [Vegan]' if i['is_vegan'] else ''}"
        f" — {i['description'] or ''} | tags: {' '.join(json.loads(i['tags'])[:4])}"
        for i in items
    ])

    # Load FULL session history so LangChain can resolve back-references
    history_rows = db.execute(
        "SELECT role, content FROM chat_messages "
        "WHERE session_id=? ORDER BY created_at ASC",
        (session_id,)
    ).fetchall()
    msgs = [{'role': r['role'], 'content': r['content']} for r in history_rows]
    msgs.append({'role': 'user', 'content': query})

    # Get Mia's reply — pass venue_id so the search_menu tool works
    reply = get_ai_chat(msgs, customer_name, menu_ctx, venue_id=vid)

    # Persist both turns to DB
    for role, content in [('user', query), ('assistant', reply)]:
        db.execute(
            "INSERT INTO chat_messages (id,venue_id,session_id,role,content,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), vid, session_id, role, content, now_str())
        )

    dishes = find_dishes(query, vid, top_k=3)
    db.commit()
    return jsonify({'session_id': session_id, 'reply': reply, 'suggested_dishes': dishes})

@app.route('/api/venues/<vid>/orders', methods=['POST'])
def place_order(vid):
    data = request.json or {}
    if not data.get('order_type') or not data.get('items'):
        return jsonify({'error': 'order_type and items required'}), 400
    db = get_db()
    if not db.execute("SELECT id FROM venues WHERE id=?", (vid,)).fetchone():
        return jsonify({'error': 'Venue not found'}), 404

    now = datetime.now()
    oid = str(uuid.uuid4())
    dietary = data.get('dietary_pref', [])
    if isinstance(dietary, str): dietary = [dietary]

    total = 0
    order_items_list = []
    for it in data['items']:
        row = db.execute("SELECT id,name,price FROM menu_items WHERE id=? AND venue_id=? AND is_available=1",
                         (it['id'], vid)).fetchone()
        if not row: return jsonify({'error': f"Item not found: {it['id']}"}), 400
        qty = max(1, int(it.get('quantity', 1)))
        total += float(row['price']) * qty
        order_items_list.append({'id': row['id'], 'name': row['name'], 'price': float(row['price']), 'qty': qty})

    db.execute("""INSERT INTO orders
        (id,venue_id,customer_name,table_ref,order_type,spice_level,dietary_pref,
         portion_size,special_instructions,total_amount,status,created_at,hour_of_day,day_of_week)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (oid, vid, data.get('customer_name','Guest'), data.get('table_ref',''),
         data['order_type'], data.get('spice_level','medium'), json.dumps(dietary),
         data.get('portion_size','regular'), data.get('special_instructions',''),
         round(total,2), 'pending', now.strftime('%Y-%m-%d %H:%M:%S'), now.hour, now.strftime('%a')))

    for it in order_items_list:
        db.execute("INSERT INTO order_items (id,order_id,menu_item_id,name,price,quantity) VALUES (?,?,?,?,?,?)",
                   (str(uuid.uuid4()), oid, it['id'], it['name'], it['price'], it['qty']))
    db.commit()

    return jsonify({'order_id': oid, 'total': round(total,2), 'status': 'pending',
                    'message': f"Order #{oid[:8].upper()} placed! We're on it ☕"}), 201

# ═══════════════════════════════════════════════
# ADMIN ROUTES
# ═══════════════════════════════════════════════

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json or {}
    if data.get('password') == ADMIN_PASSWORD:
        return jsonify({'token': ADMIN_PASSWORD, 'ok': True})
    return jsonify({'error': 'Wrong password'}), 401

@app.route('/api/admin/venues', methods=['GET'])
def admin_venues():
    err = require_admin()
    if err: return err
    rows = get_db().execute("SELECT * FROM venues ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/admin/venues', methods=['POST'])
def admin_add_venue():
    err = require_admin()
    if err: return err
    data = request.json or {}
    if not data.get('name') or not data.get('type'):
        return jsonify({'error': 'name and type required'}), 400
    vid = str(uuid.uuid4())
    get_db().execute("INSERT INTO venues (id,name,type,description,address,created_at) VALUES (?,?,?,?,?,?)",
        (vid, data['name'], data['type'], data.get('description',''), data.get('address',''), now_str()))
    get_db().commit()
    return jsonify({'id': vid}), 201

@app.route('/api/admin/venues/<vid>', methods=['DELETE'])
def admin_delete_venue(vid):
    err = require_admin()
    if err: return err
    get_db().execute("DELETE FROM venues WHERE id=?", (vid,))
    get_db().commit()
    return jsonify({'ok': True})

@app.route('/api/admin/venues/<vid>/categories', methods=['GET'])
def admin_get_categories(vid):
    err = require_admin()
    if err: return err
    rows = get_db().execute("SELECT * FROM categories WHERE venue_id=? ORDER BY sort_order", (vid,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/admin/venues/<vid>/categories', methods=['POST'])
def admin_add_category(vid):
    err = require_admin()
    if err: return err
    data = request.json or {}
    cid = str(uuid.uuid4())
    db = get_db()
    db.execute("INSERT INTO categories (id,venue_id,name,sort_order) VALUES (?,?,?,?)",
               (cid, vid, data.get('name','New Category'), data.get('sort_order',0)))
    db.commit()
    return jsonify({'id': cid}), 201

@app.route('/api/admin/categories/<cid>', methods=['DELETE'])
def admin_delete_category(cid):
    err = require_admin()
    if err: return err
    db = get_db()
    db.execute("DELETE FROM categories WHERE id=?", (cid,))
    db.commit()
    return jsonify({'ok': True})

@app.route('/api/admin/venues/<vid>/items', methods=['GET'])
def admin_get_items(vid):
    err = require_admin()
    if err: return err
    rows = get_db().execute(
        "SELECT m.*,c.name as category_name FROM menu_items m LEFT JOIN categories c ON m.category_id=c.id WHERE m.venue_id=? ORDER BY c.sort_order,m.name",
        (vid,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['tags'] = json.loads(d['tags'])
        d['is_veg'] = bool(d['is_veg'])
        d['is_vegan'] = bool(d['is_vegan'])
        d['is_available'] = bool(d['is_available'])
        result.append(d)
    return jsonify(result)

@app.route('/api/admin/venues/<vid>/items', methods=['POST'])
def admin_add_item(vid):
    err = require_admin()
    if err: return err
    data = request.json or {}
    if not data.get('name') or data.get('price') is None:
        return jsonify({'error': 'name and price required'}), 400
    iid = str(uuid.uuid4())
    get_db().execute("""INSERT INTO menu_items
        (id,venue_id,category_id,name,description,price,is_veg,is_vegan,is_available,tags,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (iid, vid, data.get('category_id'), data['name'], data.get('description',''),
         float(data['price']), int(data.get('is_veg',0)), int(data.get('is_vegan',0)),
         1, json.dumps(data.get('tags',[])), now_str()))
    get_db().commit()
    return jsonify({'id': iid}), 201

@app.route('/api/admin/items/<iid>', methods=['PUT'])
def admin_update_item(iid):
    err = require_admin()
    if err: return err
    data = request.json or {}
    db = get_db()
    db.execute("""UPDATE menu_items SET name=?,description=?,price=?,is_veg=?,is_vegan=?,
                  is_available=?,tags=?,category_id=? WHERE id=?""",
               (data.get('name'), data.get('description',''), float(data.get('price',0)),
                int(data.get('is_veg',0)), int(data.get('is_vegan',0)),
                int(data.get('is_available',1)), json.dumps(data.get('tags',[])),
                data.get('category_id'), iid))
    db.commit()
    return jsonify({'ok': True})

@app.route('/api/admin/items/<iid>', methods=['DELETE'])
def admin_delete_item(iid):
    err = require_admin()
    if err: return err
    get_db().execute("DELETE FROM menu_items WHERE id=?", (iid,))
    get_db().commit()
    return jsonify({'ok': True})

# ── TABLE MANAGEMENT ──────────────────────────
@app.route('/api/admin/venues/<vid>/tables', methods=['GET'])
def get_tables(vid):
    err = require_admin()
    if err: return err
    rows = get_db().execute(
        "SELECT id, label, sort_order FROM tables WHERE venue_id=? ORDER BY sort_order",
        (vid,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/admin/venues/<vid>/tables', methods=['POST'])
def create_table(vid):
    err = require_admin()
    if err: return err
    data = request.json or {}
    label = (data.get('label') or '').strip()
    if not label:
        return jsonify({'error': 'label required'}), 400
    db = get_db()
    max_order = db.execute("SELECT COALESCE(MAX(sort_order),0) FROM tables WHERE venue_id=?", (vid,)).fetchone()[0]
    tid = str(uuid.uuid4())
    db.execute("INSERT INTO tables (id, venue_id, label, sort_order) VALUES (?,?,?,?)",
               (tid, vid, label, max_order + 1))
    db.commit()
    return jsonify({'id': tid, 'label': label, 'sort_order': max_order + 1})

@app.route('/api/admin/venues/<vid>/tables/bulk', methods=['POST'])
def bulk_create_tables(vid):
    """Create tables Table 1 … Table N in one call."""
    err = require_admin()
    if err: return err
    data = request.json or {}
    count = int(data.get('count', 0))
    prefix = (data.get('prefix') or 'Table').strip()
    if count < 1 or count > 100:
        return jsonify({'error': 'count must be 1–100'}), 400
    db = get_db()
    # Remove existing tables first
    db.execute("DELETE FROM tables WHERE venue_id=?", (vid,))
    created = []
    for i in range(1, count + 1):
        tid = str(uuid.uuid4())
        label = f"{prefix} {i}"
        db.execute("INSERT INTO tables (id, venue_id, label, sort_order) VALUES (?,?,?,?)",
                   (tid, vid, label, i))
        created.append({'id': tid, 'label': label, 'sort_order': i})
    db.commit()
    return jsonify(created)

@app.route('/api/admin/tables/<tid>', methods=['DELETE'])
def delete_table(tid):
    err = require_admin()
    if err: return err
    get_db().execute("DELETE FROM tables WHERE id=?", (tid,))
    get_db().commit()
    return jsonify({'ok': True})

# ── ORDERS ────────────────────────────────────
@app.route('/api/admin/venues/<vid>/orders', methods=['GET'])
def admin_orders(vid):
    err = require_admin()
    if err: return err
    db = get_db()
    status_filter = request.args.get('status')
    q = "SELECT * FROM orders WHERE venue_id=?"
    params = [vid]
    if status_filter:
        q += " AND status=?"
        params.append(status_filter)
    q += " ORDER BY created_at DESC LIMIT 100"
    rows = db.execute(q, params).fetchall()
    result = []
    for r in rows:
        o = dict(r)
        o['dietary_pref'] = json.loads(o['dietary_pref'])
        items = db.execute("SELECT name,price,quantity FROM order_items WHERE order_id=?", (r['id'],)).fetchall()
        o['items'] = [dict(i) for i in items]
        result.append(o)
    return jsonify(result)

@app.route('/api/admin/orders/<oid>/status', methods=['PUT'])
def admin_update_order(oid):
    err = require_admin()
    if err: return err
    data = request.json or {}
    get_db().execute("UPDATE orders SET status=? WHERE id=?", (data.get('status'), oid))
    get_db().commit()
    return jsonify({'ok': True})

@app.route('/api/admin/venues/<vid>/analytics', methods=['GET'])
def analytics(vid):
    err = require_admin()
    if err: return err
    db = get_db()

    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    date_filter = ""
    date_params_base = [vid]
    if from_date and to_date:
        date_filter = " AND DATE(created_at) BETWEEN ? AND ?"
        date_params_base = [vid, from_date, to_date]
    elif from_date:
        date_filter = " AND DATE(created_at) >= ?"
        date_params_base = [vid, from_date]
    elif to_date:
        date_filter = " AND DATE(created_at) <= ?"
        date_params_base = [vid, to_date]

    hourly = db.execute(f"SELECT hour_of_day, COUNT(*) as count, ROUND(SUM(total_amount),2) as revenue FROM orders WHERE venue_id=?{date_filter} GROUP BY hour_of_day ORDER BY hour_of_day", date_params_base).fetchall()
    dow = db.execute(f"SELECT day_of_week, COUNT(*) as count FROM orders WHERE venue_id=?{date_filter} GROUP BY day_of_week", date_params_base).fetchall()
    top_items = db.execute(f"""SELECT oi.name, SUM(oi.quantity) as qty, ROUND(SUM(oi.price*oi.quantity),2) as revenue
        FROM order_items oi JOIN orders o ON oi.order_id=o.id WHERE o.venue_id=?{date_filter.replace('created_at', 'o.created_at')}
        GROUP BY oi.name ORDER BY qty DESC LIMIT 10""", date_params_base).fetchall()
    spice = db.execute(f"SELECT spice_level, COUNT(*) as count FROM orders WHERE venue_id=?{date_filter} AND spice_level IS NOT NULL GROUP BY spice_level", date_params_base).fetchall()
    portions = db.execute(f"SELECT portion_size, COUNT(*) as count FROM orders WHERE venue_id=?{date_filter} AND portion_size IS NOT NULL GROUP BY portion_size", date_params_base).fetchall()
    otype = db.execute(f"SELECT order_type, COUNT(*) as count FROM orders WHERE venue_id=?{date_filter} GROUP BY order_type", date_params_base).fetchall()
    dietary_raw = db.execute(f"SELECT dietary_pref FROM orders WHERE venue_id=?{date_filter}", date_params_base).fetchall()
    diet_counts = {}
    for row in dietary_raw:
        for d in json.loads(row['dietary_pref']):
            diet_counts[d] = diet_counts.get(d,0) + 1
    stats = db.execute(f"""SELECT COUNT(*) as total_orders,
        SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status='preparing' THEN 1 ELSE 0 END) as preparing,
        ROUND(AVG(total_amount),2) as avg_order_value,
        ROUND(SUM(total_amount),2) as total_revenue
        FROM orders WHERE venue_id=?{date_filter}""", date_params_base).fetchone()

    return jsonify({
        'stats': dict(stats),
        'hourly': [dict(r) for r in hourly],
        'day_of_week': [dict(r) for r in dow],
        'top_items': [dict(r) for r in top_items],
        'spice_prefs': [dict(r) for r in spice],
        'portion_prefs': [dict(r) for r in portions],
        'order_types': [dict(r) for r in otype],
        'dietary_prefs': sorted(diet_counts.items(), key=lambda x: -x[1]),
    })

# ── PAYPAL ROUTES ──────────────────────────────

@app.route('/api/paypal/create-order', methods=['POST'])
def paypal_create_order():
    data = request.json or {}
    amount = data.get('amount', 0)
    currency = data.get('currency', 'USD')
    token = get_paypal_token()
    if not token:
        return jsonify({'error': 'PayPal not configured. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET env vars.'}), 503
    try:
        r = http_requests.post(
            f'{PAYPAL_BASE}/v2/checkout/orders',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'},
            json={
                'intent': 'CAPTURE',
                'purchase_units': [{'amount': {'currency_code': currency, 'value': f'{float(amount):.2f}'}}]
            },
            timeout=15
        )
        return jsonify(r.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/paypal/capture-order/<paypal_order_id>', methods=['POST'])
def paypal_capture_order(paypal_order_id):
    token = get_paypal_token()
    if not token:
        return jsonify({'error': 'PayPal not configured'}), 503
    try:
        r = http_requests.post(
            f'{PAYPAL_BASE}/v2/checkout/orders/{paypal_order_id}/capture',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'},
            timeout=15
        )
        return jsonify(r.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    # Return an empty 204 so browsers stop logging 404 errors
    return '', 204

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    target = os.path.join(_HERE, path)
    if path and os.path.isfile(target):
        return send_from_directory(_HERE, path)
    return send_from_directory(_HERE, 'index.html')

if __name__ == '__main__':
    init_db()
    print("☕ aosa Bakehouse & Roastery — starting on http://localhost:5555")
    print(f"🔑 Admin password: {ADMIN_PASSWORD}")
    print(f"🤖 Gemini: {'✅ ready' if GOOGLE_API_KEY else '⚠️  set GOOGLE_API_KEY'}")
    print(f"💳 PayPal: {'✅ configured' if PAYPAL_CLIENT_ID != 'YOUR_PAYPAL_CLIENT_ID_HERE' else '⚠️  set PAYPAL_CLIENT_ID & PAYPAL_CLIENT_SECRET'}")
    app.run(debug=True, port=5555)