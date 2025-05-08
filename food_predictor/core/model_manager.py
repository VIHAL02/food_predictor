# Add these imports at the top of model_manager.py if not already present
import time
import gc
from datetime import datetime
import random
import os
import numpy as np
import pandas as pd
import logging
import warnings
import math
import requests
import json
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.menu_analyzer import MenuAnalyzer
from food_predictor.data.item_matcher import FoodItemMatcher
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.utils.quantity_utils import extract_quantity_value,extract_unit,validate_unit,infer_default_unit

# Disable specific warnings that cloud important logs
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages machine learning models for food quantity prediction with comprehensive 
    dependency handling and multi-source prediction capabilities.
    
    Key responsibilities:
    1. Training category and item-level XGBoost regressors
    2. Making predictions with calibrated weighting between ML and rule-based approaches
    3. Evaluating model accuracy with detailed metrics
    """
    
    def __init__(self, feature_engineer=None, item_service=None, food_rules=None, menu_analyzer=None):
        """
        Initialize model infrastructure with separate model trees for categories and items.
        
        Args:
            feature_engineer: Optional pre-configured feature engineering component
            item_service: Optional pre-configured item service component
            food_rules: Optional pre-configured food rules component
            menu_analyzer: Optional pre-configured menu analyzer component
        """
        self.category_models = {}       # Category-specific regression models
        self.item_models = {}           # Item-specific regression models
        self.category_scalers = {}      # Scalers for each category's feature space
        self.item_scalers = {}          # Scalers for each item's feature space
        self.feature_engineer = feature_engineer  # Will be injected if not provided
        self.item_service = item_service  # Will be injected if not provided
        self.food_rules = food_rules  # Will be injected if not provided
        self.guest_scaler = None        # For scaling guest count consistently
        self.menu_analyzer = menu_analyzer
        
        # Calibration config defines how to balance ML vs rule-based predictions
        self.calibration_config = self._initialize_calibration_config()
    def _initialize_calibration_config(self):
        """
        Initialize the calibration configuration that defines the weighting 
        between ML models and rule-based predictions for each category.
        
        Returns:
            dict: Comprehensive calibration configuration
        """
        return {
            "global_default": {"ml": 0.2, "rule": 0.8},
            "categories": {
                # Main Course & Complex Items
                "Biryani": {"ml": 0.2, "rule": 0.8},  # Strong historical patterns
                "Curries": {"ml": 0.2, "rule": 0.8},  # Good historical data
                "Rice": {"ml": 0.2, "rule": 0.8},  # Standard, consistent patterns
                "Flavored_Rice": {"ml": 0.2, "rule": 0.8},  # Good patterns
                "Fried_Rice": {"ml": 0.2, "rule": 0.8},  # Good patterns

                # Appetizers & Sides
                "Appetizers": {"ml": 0.2, "rule": 0.8},  # Varied consumption
                "Fried_Items": {"ml": 0.2, "rule": 0.8},  # Similar to appetizers
                "fry": {"ml": 0.2, "rule": 0.8},  # Similar to appetizers
                "Salad": {"ml": 0.2, "rule": 0.8},  # Balanced approach
                "Salan": {"ml": 0.1, "rule": 0.9},  # Balanced approach
                "Raitha": {"ml": 0.1, "rule": 0.9},  # Balanced approach

                # Soups & Liquids
                "Soups": {"ml": 0.55, "rule": 0.45},  # Fairly consistent
                "Liquids(Less_Dense)": {"ml": 0.5, "rule": 0.5},  # Complex rules
                "Liquids(High_Dense)": {"ml": 0.5, "rule": 0.5},  # Complex rules
                "Dal": {"ml": 0.55, "rule": 0.45},  # Fairly predictable

                # Condiments & Accompaniments - Rule-heavy
                "Chutneys": {"ml": 0.3, "rule": 0.7},  # Rule-driven
                "Podi": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Ghee": {"ml": 0.2, "rule": 0.8},  # Very standardized
                "Pickle": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Roti_Pachadi": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Curd": {"ml": 0.4, "rule": 0.6},  # Fairly standardized
                "Crispers": {"ml": 0.3, "rule": 0.7},  # Standard serving
                "Fryums": {"ml": 0.3, "rule": 0.7},  # Standard serving

                # Bread items - Rule-heavy
                "Breads": {"ml": 0.3, "rule": 0.7},  # Usually piece-based
                "Paan": {"ml": 0.2, "rule": 0.8},  # Standard serving

                # Breakfast items
                "Breakfast": {"ml": 0.5, "rule": 0.5},  # Varied patterns
                "Sandwich": {"ml": 0.5, "rule": 0.5},  # Standard portion
                "Papad": {"ml": 0.4, "rule": 0.6},  # Usually piece-based

                # Desserts & Sweet items
                "Desserts": {"ml": 0.1, "rule": 0.9},  # Rule-driven
                "Cakes": {"ml": 0.1, "rule": 0.9},  # Size-dependent
                "Cup_Cakes": {"ml": 0.3, "rule": 0.7},  # Standard portion
                "Fruits": {"ml": 0.5, "rule": 0.5},  # Fairly standard

                # Beverages
                "Welcome_Drinks": {"ml": 0.4, "rule": 0.6},  # Standard serving
                "Hot_Beverages": {"ml": 0.4, "rule": 0.6},  # Standard serving

                # International categories
                "Italian": {"ml": 0.5, "rule": 0.5},  # Balanced approach
                "Pizza": {"ml": 0.5, "rule": 0.5}  # Balanced approach
            },
            "confidence_thresholds": {
                "high": {"threshold": 0.8, "ml": 0.7, "rule": 0.3},
                "medium": {"threshold": 0.5, "ml": 0.5, "rule": 0.5},
                "low": {"threshold": 0.0, "ml": 0.3, "rule": 0.7}
            }
        }
        
    def fit(self, data):
        """
        Train category and item-level XGBoost regressors with dependency-aware features.
        
        Args:
            data (DataFrame): Training data with order details and quantities
            food_rules (FoodCategoryRules): Food category rules component
            feature_engineer (FeatureEngineer): Feature engineering component
            item_matcher (FoodItemMatcher): Food item matching component
            
        Returns:
            self: Trained model manager instance
        """
        logger.info("Starting model training")
        self.feature_engineer 
        
        # Ensure required columns are present
        assert 'Event_Time' in data.columns and 'Meal_Time' in data.columns and 'Event_Type' in data.columns, \
            "Data must contain 'Event_Time', 'Meal_Time', and 'Event_Type' columns for encoding"

        # 1) Fit feature engineering components
        self.feature_engineer.fit_encoders(data)

        # 2) Ensure guest count column consistency
        data = data.copy()
        if 'total_guest_count' not in data.columns and 'Guest_Count' in data.columns:
            data['total_guest_count'] = data['Guest_Count']
        elif 'Guest_Count' not in data.columns and 'total_guest_count' in data.columns:
            data['Guest_Count'] = data['total_guest_count']

        # 3) Build per-order contexts and extract rich menu features
        logger.debug("Building per-order menu contexts and features")
        order_menu_features = {}
        
        for i, (order_number, order_data) in enumerate(data.groupby('Order_Number')):
            if i < 5:
                logger.debug(f"[Order {order_number}] items: {order_data['Item_name'].tolist()}")
                
            # Extract order metadata
            evt_time = order_data['Event_Time'].iat[0]
            meal_t = order_data['Meal_Time'].iat[0]
            evt_type = order_data['Event_Type'].iat[0]
            guests = order_data['total_guest_count'].iat[0]
            vegs = order_data['Veg_Count'].iat[0] if 'Veg_Count' in order_data else int(0.4 * guests)
            items = order_data['Item_name'].unique().tolist()

            # Build comprehensive menu context
            menu_context = self.menu_analyzer.build_menu_context(
                evt_time, meal_t, evt_type,
                guests, vegs, items
            )
            
            # Extract rich menu features including all dependency-aware features
            feats = self.feature_engineer.extract_menu_features(
                data=order_data, 
                menu_context=menu_context,
                food_rules = self.food_rules
            )
            
            if i < 5:
                logger.debug(f"[Order {order_number}] extracted features: {feats}")
                
            order_menu_features[order_number] = feats

        # 4) Flatten feature dicts into DataFrame and merge back
        mf_df = (
            pd.DataFrame.from_dict(order_menu_features, orient='index')
            .reset_index()
            .rename(columns={'index': 'Order_Number'})
        )
        data = data.merge(mf_df, on='Order_Number', how='left')

        logger.debug("Merged dependency features into data; sample rows:\n%s", 
                     data.head().to_dict(orient='records'))

        # 5) Prepare target variables
        data['quantity_value'] = data['Per_person_quantity'].apply(extract_quantity_value)
        
        # 6) Generate feature visualization dataset for debugging
        X = self.feature_engineer.prepare_features(data)
        sample_features = self.feature_engineer.print_feature_engineering_dataset(data)
        logger.info("Feature engineering visualization completed")
        y = data['quantity_value']
        # 7) Train category-level models
        for category in data['Category'].unique():
            cat_df = data[data['Category'] == category]
            if cat_df.empty:
                continue
                
            logger.debug(f"Training category model for '{category}' on {len(cat_df)} rows")
            
            # Aggregate quantities at the order level for consistent training
            grp = (
                cat_df.groupby('Order_Number')
                    .agg(per_person_quantity=('quantity_value','sum'),
                         total_guests=('total_guest_count','first'))
                    .reset_index()
            )
            
            train_df = grp.merge(
                data[['Order_Number','Event_Time','Meal_Time','Event_Type']].drop_duplicates(),
                on='Order_Number', how='left'
            )

            # Prepare features using the feature engineer
            X_cat = self.feature_engineer.prepare_features(train_df)
            logger.debug(f"  -> X_cat.shape={X_cat.shape}")
            y_cat = train_df['per_person_quantity']
            n_samples = X_cat.shape[0]
            
            # Initialize and train category model
            if category not in self.category_models:
                self.category_scalers[category] = RobustScaler()
                Xc_scaled = self.category_scalers[category].fit_transform(X_cat)
                
                # Configure XGBoost model with carefully chosen hyperparameters
                
                model = XGBRegressor(
                        n_estimators=800, 
                        learning_rate=0.05,
                        max_depth=5, 
                        min_child_weight=2,
                        gamma=0.1,          # Add from item model
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.5,      # Add from item model
                        reg_lambda=1.0,     # Add from item model
                        objective='reg:squarederror',
                        eval_metric='rmse', # Add from item model
                        random_state=42
                    )
                                    
                # For very small sample sizes, skip CV
                if n_samples < 3:
                    logger.warning(f"Category '{category}' has only {n_samples} samples, skipping CV.")
                    model.fit(Xc_scaled, y_cat)
                    self.category_models[category] = model
                    continue
                
                # Fit model on all data directly
                model.fit(Xc_scaled, y_cat)
                self.category_models[category] = model
            else:
                # Update existing model
                Xc_scaled = self.category_scalers[category].transform(X_cat)
                self.category_models[category].fit(Xc_scaled, y_cat)
        
        # 8) Train item-level models
        for item in data['Item_name'].unique():
            itm_df = data[data['Item_name'] == item]
            if itm_df.empty:
                continue
                
            logger.debug(f"Training item model for '{item}' on {len(itm_df)} rows")
            
            # Aggregate quantities at the order level
            grp = (
                itm_df.groupby('Order_Number')
                    .agg(per_person_quantity=('quantity_value','sum'),
                         total_guests=('total_guest_count','first'))
                    .reset_index()
            )
            
            train_df = grp.merge(
                data[['Order_Number','Event_Time','Meal_Time','Event_Type']].drop_duplicates(),
                on='Order_Number', how='left'
            )

            # Prepare features
            X_itm = self.feature_engineer.prepare_features(train_df)
            logger.debug(f"  -> X_item.shape={X_itm.shape}")
            y_itm = train_df['per_person_quantity']
            
            # Initialize and train item model
            if item not in self.item_models:
                self.item_scalers[item] = RobustScaler()
                Xi_scaled = self.item_scalers[item].fit_transform(X_itm)
                
                # Item models use more complex hyperparameters to capture item-specific patterns
                model = XGBRegressor(
                    n_estimators=1000, 
                    learning_rate=0.01,
                    max_depth=10, 
                    min_child_weight=2,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    random_state=42
                )
                
                # Fit model directly on all data
                model.fit(Xi_scaled, y_itm)
                self.item_models[item] = model
            else:
                # Update existing model
                Xi_scaled = self.item_scalers[item].transform(X_itm)
                self.item_models[item].fit(Xi_scaled, y_itm)

        logger.info(f"Model training completed: {len(self.category_models)} category models, "
                   f"{len(self.item_models)} item models")
        return self
    
   
    
    def predict(self, event_time, meal_type, event_type, total_guest_count, veg_guest_count=None, selected_items= None):
        """
        Multi-source quantity prediction with comprehensive dependency handling.
        
        Args:
            event_time (str): Time of the event
            meal_type (str): Type of meal
            event_type (str): Type of event
            total_guest_count (int): Total number of guests
            veg_guest_count (int): Number of vegetarian guests
            selected_items (list): List of selected menu items
            food_rules (FoodCategoryRules): Food category rules
            feature_engineer (FeatureEngineer): Feature engineering component
            
        Returns:
            dict: Detailed predictions with multiple prediction sources
        """
        # Preprocessing and context initialization
        if veg_guest_count is None:
            veg_guest_count = 0
    
        logger.debug(f"→ predict(): event_time={event_time!r}, meal_type={meal_type!r}, "
                     f"event_type={event_type!r}, total_guests={total_guest_count}, "
                     f"veg_guests={veg_guest_count}, items={selected_items!r}")
        
        # Save reference to feature engineer if not set
        self.feature_engineer
            
        # 1) Build comprehensive menu context
        menu_context = self.menu_analyzer.build_menu_context(
            event_time, meal_type, event_type,
            total_guest_count, veg_guest_count, selected_items
        )
        logger.debug(f"   menu_context built, categories={menu_context['categories']}")
        
        # 2) Create input data structure
        n = len(selected_items)
        input_data = pd.DataFrame({
            'Event_Time':        [event_time] * n,
            'Meal_Time':         [meal_type] * n,
            'Event_Type':        [event_type] * n,
            'total_guest_count': [total_guest_count] * n,
            'Veg_Count':         [veg_guest_count] * n,
            'Item_name':         selected_items
        })
        
        # 3) Extract rich menu features
        menu_features = self.feature_engineer.extract_menu_features(
            data=input_data, 
            menu_context=menu_context,
            food_rules=self.food_rules
        )
        logger.debug(f"   menu_features = {menu_features}")
      
        # 4) Add menu features to input data (avoiding DataFrame fragmentation)
        menu_features_df = pd.DataFrame(
            {feat_name: [feat_val] * len(input_data) 
             for feat_name, feat_val in menu_features.items()}, 
            index=input_data.index
        )
        input_data = pd.concat([input_data, menu_features_df], axis=1)

        logger.debug(
            "   input_data with flattened features:\n%s",
            input_data.head().to_dict(orient='records')
        )
                
        # 5) Prepare feature matrix using the feature engineer
        X = self.feature_engineer.prepare_features(input_data)
        logger.debug(f"   feature matrix X.shape = {X.shape}")
        
        # 6) Identify main items using Gemini API
        main_items_info = self.menu_analyzer.identify_main_items(
            menu_context, 
            menu_context['meal_type'], 
            menu_context['categories']
        )
        logger.debug(f"   main_items_info = {main_items_info}")
        
        # 7) Initialize prediction tracking structures
        predictions = {}
        prediction_details = {}
        category_quantities = {}
        category_per_person = {}
        item_predictions = {}
        items_by_category = menu_context['items_by_category']
        
        # Non-vegetarian guest count
        non_veg_guest_count = total_guest_count - veg_guest_count
        
        # 8) Per-item prediction with multi-source approach
        for item in selected_items:
            # Item property extraction
            item_properties = menu_context['item_properties'][item]
            category = item_properties['category']
            
            # Get calibration weights for this category
            if category in self.calibration_config['categories']:
                calibration = self.calibration_config['categories'][category]
            else:
                calibration = self.calibration_config['global_default']
            
            ml_weight = calibration['ml']
            rule_weight = calibration['rule']
            
            # Multi-source prediction
            category_model_pred = None
            item_model_pred = None
            calibrated_category_pred = None
            calibrated_item_pred = None
            
            # Category-level ML prediction
            if category in self.category_models:
                X_cat_scaled = self.category_scalers[category].transform(X)
                category_model_pred = float(self.category_models[category].predict(X_cat_scaled)[0])
            
            # Item-level ML prediction
            if item in self.item_models:
                X_item_scaled = self.item_scalers[item].transform(X)
                item_model_pred = float(self.item_models[item].predict(X_item_scaled)[0])
            
            # Rule-based prediction
            category_rules_result = self.food_rules.apply_category_rules(
                item, item_properties, menu_context, main_items_info
            )
            category_rules_pred = category_rules_result['quantity']
            
            # Apply calibration weights to blend predictions
            if category_model_pred is not None:
                calibrated_category_pred = (category_rules_pred * rule_weight) + (ml_weight * category_model_pred)
            if item_model_pred is not None:
                calibrated_item_pred = (category_rules_pred * rule_weight) + (ml_weight * item_model_pred)
            
            logger.debug(
                f"   raw preds for {item!r}: cat_ml={category_model_pred}, item_ml={item_model_pred}, "
                f"rules={category_rules_pred}, cal_cat={calibrated_category_pred}, cal_item={calibrated_item_pred}"
            )

            # Store detailed item prediction data
            item_predictions[item] = {
                'category': category,
                'is_veg': item_properties.get('is_veg', 'veg'),
                'unit': category_rules_result['unit'],
                'predictions': {
                    'category_model': category_model_pred,
                    'item_model': item_model_pred,
                    'rules_based': category_rules_pred,
                    'calibrated_category': calibrated_category_pred,
                    'calibrated_item': calibrated_item_pred
                },
                'calibration_used': calibration,
                'properties': item_properties
            }
            
            # Base quantity from rules
            per_person_quantity = category_rules_pred
            
            # Categories that need vegetarian-specific handling
            veg_specific_categories = [
                "Biryani", 
                "Appetizers", 
                "Curries", 
                "Flavored_Rice", 
                "Fried_Rice", 
                "Italian"
            ]
            
            # Extract vegetarian status
            is_veg = item_properties.get('is_veg', 'veg')
            
            # Select best available prediction source
            selected_quantity = item_model_pred if item_model_pred is not None else category_model_pred

            if selected_quantity is None:
                logger.warning(f"   No prediction found for {item}, using default quantity")
                selected_quantity = category_rules_pred 
            
            # Calculate total quantity based on guest count
            total_quantity = selected_quantity * total_guest_count
            
            # Apply veg/non-veg guest ratio adjustments
            if veg_guest_count != 0:
                total_quantity = selected_quantity * non_veg_guest_count

                # Specific veg categories recalculation
                if (category in veg_specific_categories) and (is_veg == 'veg'):
                    logger.debug(f"Applying veg-specific logic for {item}, category={category}, is_veg={is_veg}")
                    selected_quantity = item_model_pred or category_model_pred 
                    logger.debug(f"Changing from {selected_quantity} to {item_model_pred or category_model_pred }")
                    total_quantity = selected_quantity * veg_guest_count
            
            # Store category quantities
            if category not in category_quantities:
                category_quantities[category] = {
                    "value": total_quantity,
                    "unit": category_rules_result['unit']
                }
                category_per_person[category] = per_person_quantity
            else:
                category_quantities[category]["value"] += total_quantity
                
            # Detailed prediction information
            prediction_details = {
                'ml_predictions': {
                    'category_model': {
                        'quantity': category_model_pred,
                        'available': category_model_pred is not None,
                        'calibrated': calibrated_category_pred
                    },
                    'item_model': {
                        'quantity': item_model_pred,
                        'available': item_model_pred is not None,
                        'calibrated': calibrated_item_pred
                    }
                },
                'category_rules_prediction': {
                    'quantity': category_rules_pred
                },
                'calibration_used': calibration
            }
      
            # Store final prediction
            predictions[item] = { 
                'total': f"{total_quantity:.1f}{item_predictions[item]['unit']}", 
                'per_person': f"{selected_quantity:.2f}{item_predictions[item]['unit']}", 
                'unit': item_predictions[item]['unit'], 
                'category': category, 
                'is_veg': is_veg, 
                'prediction_details': prediction_details,
                'selected_prediction_source': 'item_model' if item_model_pred is not None else 
                                             ('category_model' if category_model_pred is not None else 'rules_based')
            }
                    
        # 9) Apply meal type modifiers for final adjustments
        for category in category_per_person:
            orig_qty = category_per_person[category]
            modified_qty = self.food_rules.apply_meal_type_modifier(meal_type, orig_qty)
            
            category_per_person[category] = modified_qty
            category_quantities[category]["value"] = modified_qty
            
        logger.debug(f" predict() final output = {predictions}")
        return predictions
    
    
    
    def save_models(self, model_dir):
        """
        Serialize complete ML pipeline including preprocessing state.
        
        Args:
            model_dir (str): Output file path
        """
        import os
        import joblib
        
        # Handle path resolution
        if os.path.isdir(model_dir) or model_dir.endswith('/') or model_dir.endswith('\\'):
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'food_prediction_model.joblib')
        else:
            parent_dir = os.path.dirname(model_dir)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            model_path = model_dir
        
        # Include feature engineering state
        feature_engineering_state = {
            'event_time_encoder': self.feature_engineer.event_time_encoder,
            'meal_type_encoder': self.feature_engineer.meal_type_encoder,
            'event_type_encoder': self.feature_engineer.event_type_encoder
        }
        
        # Include any additional stateful components
        if hasattr(self.feature_engineer, 'guest_scaler') and self.feature_engineer.guest_scaler:
            feature_engineering_state['guest_scaler'] = self.feature_engineer.guest_scaler
        
        # Optional: Include all feature engineer scalers
        for attr_name in dir(self.feature_engineer):
            if attr_name.endswith('_scaler') and hasattr(self.feature_engineer, attr_name):
                scaler = getattr(self.feature_engineer, attr_name)
                if scaler is not None:
                    feature_engineering_state[attr_name] = scaler
        
        # Complete ML pipeline state
        master_model = {
            # Feature engineering (preprocessing)
            'feature_engineering_state': feature_engineering_state,
            
            # Model components (core prediction)
            'category_models': self.category_models,
            'item_models': self.item_models,
            'category_scalers': self.category_scalers,
            'item_scalers': self.item_scalers,
            
            # Post-processing components
            'calibration_config': self.calibration_config
        }
        
        # Serialize with optimal compression
        joblib.dump(master_model, model_path, compress=3, protocol=4)
        
        print(f"Model saved to {model_path}")
        return model_path

    def load_models(self, model_path):
        """
        Load complete ML pipeline with preprocessing state.
        
        Args:
            model_path (str): Path to model file or directory
        """
        import os
        import joblib
        
        # Resolve path handling
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'food_prediction_model.joblib')
        
        # Validate existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load pipeline state
        master_model = joblib.load(model_path)
        
        # Restore feature engineering state
        feature_engineering_state = master_model.get('feature_engineering_state', {})
        for component_name, component in feature_engineering_state.items():
            setattr(self.feature_engineer, component_name, component)
        
        # Restore model components
        self.category_models = master_model['category_models']
        self.item_models = master_model['item_models']
        self.category_scalers = master_model['category_scalers']
        self.item_scalers = master_model['item_scalers']
        self.calibration_config = master_model['calibration_config']
        
        print(f"Loaded model with {len(self.category_models)} category models and "
            f"{len(self.item_models)} item models")
        
        return self
    
    def evaluate_item_accuracy(self, data, food_rules, feature_engineer, item_matcher, 
                              num_orders=100, random_state=42, output_file="evaluation_results.xlsx"):
        """
        Evaluate model accuracy on a random sample of orders and save results to Excel.
        
        Args:
            data (DataFrame): Evaluation dataset
            food_rules (FoodCategoryRules): Food category rules
            feature_engineer (FeatureEngineer): Feature engineering component
            item_matcher (FoodItemMatcher): Food item matcher
            num_orders (int): Number of orders to sample
            random_state (int): Random seed for reproducibility
            output_file (str): Excel file path for results
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        import numpy as np
        import random
        import traceback
        
        print(f"\n=== Evaluating Item-Level Accuracy on {num_orders} Random Orders ===\n")
        
        # Set random seed for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Store reference to feature engineer if not set
        if not self.feature_engineer:
            self.feature_engineer = feature_engineer
            
        # Get unique order numbers and sample randomly
        unique_orders = data['Order_Number'].unique()
        if len(unique_orders) < num_orders:
            print(f"Warning: Only {len(unique_orders)} orders available, using all of them.")
            sampled_orders = unique_orders
        else:
            sampled_orders = np.random.choice(unique_orders, size=num_orders, replace=False)
        
        print(f"Sampled orders: {sampled_orders}")
        
        # Create test dataset
        test_data = data[data['Order_Number'].isin(sampled_orders)]
        
        # Prepare for evaluation
        results = {
            'overall': {
                'predictions': [],
                'actual': [],
                'errors': []
            },
            'items': {},
            'categories': {}
        }
        
        # Create DataFrames for Excel output
        detailed_results = []
        order_summary = []
        
        # Evaluate on test orders
        print("\nEvaluating on test orders:")
        for order_num in sampled_orders:
            order_data = test_data[test_data['Order_Number'] == order_num]
            
            # Skip if order data is empty
            if order_data.empty:
                print(f"  - Warning: No data found for test order {order_num}, skipping.")
                continue
            
            # Extract order information
            evt_time = order_data['Event_Time'].iat[0]
            meal_t = order_data['Meal_Time'].iat[0]
            evt_type = order_data['Event_Type'].iat[0]
            
            # Handle different column names for guest count
            if 'total_guest_count' in order_data.columns:
                guests = order_data['total_guest_count'].iat[0]
            elif 'Guest_Count' in order_data.columns:
                guests = order_data['Guest_Count'].iat[0]
            else:
                guests = 50  # Default if neither column exists
                
            # Handle vegetarian guest count
           # Modified implementation with parameter optionality and NULL propagation
            if 'Veg_Count' in order_data.columns and not pd.isna(order_data['Veg_Count'].iat[0]):
                vegs = order_data['Veg_Count'].iat[0]
                # Optional statistical tracking
                if not hasattr(self, '_eval_stats'):
                    self._eval_stats = {'veg_provided': 0, 'veg_omitted': 0}
                self._eval_stats['veg_provided'] += 1
            else:
                # Pass None to trigger default behavior in predict()
                vegs = None
                # Optional statistical tracking
                if not hasattr(self, '_eval_stats'):
                    self._eval_stats = {'veg_provided': 0, 'veg_omitted': 0}
                self._eval_stats['veg_omitted'] += 1
            
            # Get unique items in this order
            items = order_data['Item_name'].unique().tolist()
            
            try:
                # Get predictions for these items using the trained model
                predictions = self.predict(
                    evt_time, meal_t, evt_type, 
                    guests, vegs, items
                )
                
                print(f"\nOrder: {order_num}, Items: {len(items)}")
                print(f"Event: {evt_time}, {meal_t}, {evt_type}, {guests} guests ({vegs} vegetarian)")
                
                # Track items with issues
                skipped_items = []
                
                for item in items:
                    # Skip items that aren't in the predictions
                    if item not in predictions:
                        skipped_items.append(item)
                        continue
                        
                    try:
                        item_data_rows = order_data[order_data['Item_name'] == item]
                        if item_data_rows.empty:
                            skipped_items.append(item)
                            continue
                            
                        item_data_row = item_data_rows.iloc[0]
                        
                        # Get category from predictions safely
                        if 'category' not in predictions[item]:
                            skipped_items.append(item)
                            continue
                            
                        category = predictions[item]['category']
                        
                        # Get actual quantity safely
                        if 'Per_person_quantity' not in item_data_row:
                            skipped_items.append(item)
                            continue
                            
                        actual_qty = extract_quantity_value(item_data_row['Per_person_quantity'])
                        actual_unit = extract_unit(item_data_row['Per_person_quantity'])
                        
                        # Skip if actual quantity couldn't be extracted
                        if actual_qty is None or actual_qty == 0:
                            skipped_items.append(item)
                            continue
                            
                        # Handle missing actual unit
                        if not actual_unit:
                            if 'per_person' in predictions[item] and predictions[item]['per_person']:
                                actual_unit = extract_unit(predictions[item]['per_person'])
                            else:
                                actual_unit = 'g'  # Default unit
                        
                        # Get predicted quantity safely
                        if 'per_person' not in predictions[item]:
                            skipped_items.append(item)
                            continue
                            
                        pred_qty = extract_quantity_value(predictions[item]['per_person'])
                        pred_unit = extract_unit(predictions[item]['per_person'])
                        
                        # Skip if predicted quantity couldn't be extracted
                        if pred_qty is None:
                            skipped_items.append(item)
                            continue
                        
                        # Ensure units match for comparison
                        if actual_unit != pred_unit:
                            # Simple unit conversion if possible
                            if actual_unit == 'kg' and pred_unit == 'g':
                                actual_qty *= 1000
                                actual_unit = 'g'
                            elif actual_unit == 'g' and pred_unit == 'kg':
                                actual_qty /= 1000
                                actual_unit = 'kg'
                            elif actual_unit == 'l' and pred_unit == 'ml':
                                actual_qty *= 1000
                                actual_unit = 'ml'
                            elif actual_unit == 'ml' and pred_unit == 'l':
                                actual_qty /= 1000
                                actual_unit = 'l'
                        
                        # Calculate error
                        error = pred_qty - actual_qty
                        error_pct = (error / actual_qty) * 100 if actual_qty != 0 else float('inf')
                        
                        # Store detailed results
                        detailed_results.append({
                            'Order_Number': order_num,
                            'Item': item,
                            'Category': category,
                            'Predicted_Quantity': pred_qty,
                            'Predicted_Unit': pred_unit,
                            'Actual_Quantity': actual_qty,
                            'Actual_Unit': actual_unit,
                            'Error': error,
                            'Error_Percentage': error_pct
                        })
                        
                        # Store results for overall analysis
                        results['overall']['predictions'].append(pred_qty)
                        results['overall']['actual'].append(actual_qty)
                        results['overall']['errors'].append(error)
                        
                        # Store item-specific results
                        if item not in results['items']:
                            results['items'][item] = {
                                'predictions': [],
                                'actual': [],
                                'errors': [],
                                'category': category
                            }
                        results['items'][item]['predictions'].append(pred_qty)
                        results['items'][item]['actual'].append(actual_qty)
                        results['items'][item]['errors'].append(error)
                        
                        # Store category-specific results
                        if category not in results['categories']:
                            results['categories'][category] = {
                                'predictions': [],
                                'actual': [],
                                'errors': []
                            }
                        results['categories'][category]['predictions'].append(pred_qty)
                        results['categories'][category]['actual'].append(actual_qty)
                        results['categories'][category]['errors'].append(error)
                        
                    except Exception as e:
                        print(f"  Error processing item {item}: {str(e)}")
                        skipped_items.append(item)
                
                # Store order summary
                order_summary.append({
                    'Order_Number': order_num,
                    'Event_Time': evt_time,
                    'Meal_Time': meal_t,
                    'Event_Type': evt_type,
                    'Total_Guests': guests,
                    'Vegetarian_Guests': 0 if vegs is None else vegs,
                    'Total_Items': len(items),
                    'Skipped_Items': len(skipped_items)
                })
            
            except Exception as e:
                print(f"Error processing order {order_num}: {str(e)}")
                print(traceback.format_exc())
        
        # Calculate overall metrics if we have data
        if not results['overall']['actual']:
            print("\nNo valid predictions to evaluate. Exiting.")
            return None
            
        overall_actual = np.array(results['overall']['actual'])
        overall_pred = np.array(results['overall']['predictions'])
        
        overall_mae = mean_absolute_error(overall_actual, overall_pred)
        overall_rmse = np.sqrt(mean_squared_error(overall_actual, overall_pred))
        overall_mape = np.mean(np.abs((overall_actual - overall_pred) / overall_actual)) * 100
        overall_r2 = r2_score(overall_actual, overall_pred)
        
        # Calculate item-level metrics
        item_metrics = []
        for item, item_data in results['items'].items():
            item_actual = np.array(item_data['actual'])
            item_pred = np.array(item_data['predictions'])
            
            item_mae = mean_absolute_error(item_actual, item_pred)
            item_rmse = np.sqrt(mean_squared_error(item_actual, item_pred))
            item_mape = np.mean(np.abs((item_actual - item_pred) / item_actual)) * 100
            item_r2 = r2_score(item_actual, item_pred)
            
            item_metrics.append({
                'Item': item,
                'Category': item_data['category'],
                'MAE': item_mae,
                'RMSE': item_rmse,
                'MAPE': item_mape,
                'R2': item_r2,
                'Count': len(item_actual)
            })
        
        # Calculate category-level metrics
        category_metrics = []
        for category, cat_data in results['categories'].items():
            cat_actual = np.array(cat_data['actual'])
            cat_pred = np.array(cat_data['predictions'])
            
            cat_mae = mean_absolute_error(cat_actual, cat_pred)
            cat_rmse = np.sqrt(mean_squared_error(cat_actual, cat_pred))
            cat_mape = np.mean(np.abs((cat_actual - cat_pred) / cat_actual)) * 100
            cat_r2 = r2_score(cat_actual, cat_pred)
            
            category_metrics.append({
                'Category': category,
                'MAE': cat_mae,
                'RMSE': cat_rmse,
                'MAPE': cat_mape,
                'R2': cat_r2,
                'Count': len(cat_actual)
            })
        
        # Create overall metrics DataFrame
        overall_metrics = pd.DataFrame([{
            'Metric': 'Overall',
            'MAE': overall_mae,
            'RMSE': overall_rmse,
            'MAPE': overall_mape,
            'R2': overall_r2,
            'Count': len(overall_actual)
        }])
        
        # Convert all results to DataFrames
        detailed_df = pd.DataFrame(detailed_results)
        order_summary_df = pd.DataFrame(order_summary)
        item_metrics_df = pd.DataFrame(item_metrics)
        category_metrics_df = pd.DataFrame(category_metrics)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_file) as writer:
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            order_summary_df.to_excel(writer, sheet_name='Order_Summary', index=False)
            item_metrics_df.to_excel(writer, sheet_name='Item_Metrics', index=False)
            category_metrics_df.to_excel(writer, sheet_name='Category_Metrics', index=False)
            overall_metrics.to_excel(writer, sheet_name='Overall_Metrics', index=False)
        
        print(f"\nResults saved to {output_file}")
        print("\n=== Overall Results ===")
        print(f"MAE: {overall_mae:.2f}")
        print(f"RMSE: {overall_rmse:.2f}")
        print(f"MAPE: {overall_mape:.2f}%")
        print(f"R²: {overall_r2:.4f}")
        
        return results
    def evaluate_on_external_test_set(self, test_data_path, output_file=None, batch_size=100):
        """
        Evaluate model on external test dataset (.xlsx) with memory-efficient batched processing.
        
        Args:
            test_data_path (str): Path to test dataset Excel file (.xlsx)
            output_file (str): Path for results (defaults to test_results_{timestamp}.xlsx)
            batch_size (int): Number of orders to process in each batch to manage memory
            
        Returns:
            dict: Comprehensive performance metrics
        """
        import os
        import time
        import gc
        from datetime import datetime
        
        # Validate file extension
        file_ext = os.path.splitext(test_data_path)[1].lower()
        if file_ext not in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {file_ext}. Only Excel files (.xlsx, .xls) are supported.")
        
        # Generate default output filename with timestamp
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_results_{timestamp}.xlsx"
        
        print(f"\n=== Evaluating Model on External Test Set (.xlsx) ===\n")
        start_time = time.time()
        
        # Memory-efficient Excel handling - first extract just the order numbers
        print("Scanning Excel dataset...")
        
        # Use ExcelFile context manager for efficient handling
        with pd.ExcelFile(test_data_path) as excel_file:
            # Get first sheet by default
            sheet_name = excel_file.sheet_names[0]
            # Extract only Order_Number column to minimize memory usage
            orders_df = pd.read_excel(
                excel_file, 
                sheet_name=sheet_name,
                usecols=['Order_Number']
            ).drop_duplicates()
            
        # Get unique order numbers and count
        test_orders = orders_df['Order_Number'].unique()
        total_orders = len(test_orders)
        print(f"Found {total_orders} unique orders in test dataset")
        
        # Initialize results structure with nested dictionaries
        results = self._initialize_results_structure()
        
        # Process in batches to manage memory
        batch_count = (total_orders + batch_size - 1) // batch_size  # Ceiling division
        print(f"Processing in {batch_count} batches of {batch_size} orders each")
        
        detailed_results = []
        order_summary = []
        
        # Track timing for ETA calculation
        processed_count = 0
        batch_start_time = time.time()
        
        # Process each batch
        for batch_idx in range(batch_count):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_orders)
            batch_orders = test_orders[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_idx+1}/{batch_count} (orders {batch_start+1}-{batch_end})")
            
            # For Excel files, we read the entire sheet then filter in memory
            # This is less memory-efficient than CSV query filters, but Excel doesn't support query loading
            batch_filter_list = batch_orders.tolist()
            
            # Read full Excel data for this batch
            batch_data = pd.read_excel(
                test_data_path,
                converters={'Order_Number': str}  # Ensure string comparison works correctly
            )
            
            # Filter post-load to only include orders in this batch
            batch_data = batch_data[batch_data['Order_Number'].isin(batch_filter_list)]
            
            # Handle column normalization
            if 'total_guest_count' not in batch_data.columns and 'Guest_Count' in batch_data.columns:
                batch_data['total_guest_count'] = batch_data['Guest_Count']
            elif 'Guest_Count' not in batch_data.columns and 'total_guest_count' in batch_data.columns:
                batch_data['Guest_Count'] = batch_data['total_guest_count']
                
            
            
            # Process each order in the batch
            for order_num in batch_data['Order_Number'].unique():
                order_data = batch_data[batch_data['Order_Number'] == order_num]
                
                # Skip if order data is empty
                if order_data.empty:
                    continue
                
                # Extract order metadata
                evt_time = order_data['Event_Time'].iat[0]
                meal_t = order_data['Meal_Time'].iat[0]
                evt_type = order_data['Event_Type'].iat[0]
                guests = order_data['total_guest_count'].iat[0]
                if 'Veg_Count' in order_data.columns and not pd.isna(order_data['Veg_Count'].iat[0]):
                    vegs = order_data['Veg_Count'].iat[0]
                    # Statistical tracking for evaluation distribution
                    if not hasattr(self, '_eval_stats'):
                        self._eval_stats = {'veg_provided': 0, 'veg_omitted': 0}
                    self._eval_stats['veg_provided'] += 1
                else:
                    # Pass None to trigger default handling in predict()
                    vegs = None
                    # Statistical tracking for evaluation distribution
                    if not hasattr(self, '_eval_stats'):
                        self._eval_stats = {'veg_provided': 0, 'veg_omitted': 0}
                    self._eval_stats['veg_omitted'] += 1

                
                # Get unique items in this order
                items = order_data['Item_name'].unique().tolist()
                
                try:
                    # Set a flag to optimize API calls during evaluation
                    self._evaluation_mode = True
                    
                    # Get predictions for these items
                    predictions = self.predict(evt_time, meal_t, evt_type, guests, vegs, items)
                    
                    # Reset flag after prediction
                    self._evaluation_mode = False
                    
                    # Count successful predictions
                    prediction_count = 0
                    skipped_items = []
                    
                    # Process each item's prediction
                    for item in items:
                        if item not in predictions:
                            skipped_items.append(item)
                            continue
                        
                        try:
                            # Process item predictions and calculate metrics
                            item_data_rows = order_data[order_data['Item_name'] == item]
                            if item_data_rows.empty:
                                skipped_items.append(item)
                                continue
                            
                            item_data = item_data_rows.iloc[0]
                            category = predictions[item]['category']
                            
                            # Extract quantities with robust error handling
                            try:
                                # Get actual quantity
                                if 'Per_person_quantity' not in item_data:
                                    skipped_items.append(item)
                                    continue
                                    
                                actual_qty = extract_quantity_value(item_data['Per_person_quantity'])
                                actual_unit = extract_unit(item_data['Per_person_quantity'])
                                
                                if actual_qty is None or actual_qty == 0:
                                    skipped_items.append(item)
                                    continue
                                    
                                # Get predicted quantity
                                if 'per_person' not in predictions[item]:
                                    skipped_items.append(item)
                                    continue
                                    
                                pred_qty = extract_quantity_value(predictions[item]['per_person'])
                                pred_unit = extract_unit(predictions[item]['per_person'])
                                
                                if pred_qty is None:
                                    skipped_items.append(item)
                                    continue
                                
                                # Handle unit conversion for comparison
                                if actual_unit != pred_unit:
                                    # Apply standard unit conversions
                                    if actual_unit == 'kg' and pred_unit == 'g':
                                        actual_qty *= 1000
                                        actual_unit = 'g'
                                    elif actual_unit == 'g' and pred_unit == 'kg':
                                        actual_qty /= 1000
                                        actual_unit = 'kg'
                                    elif actual_unit == 'l' and pred_unit == 'ml':
                                        actual_qty *= 1000
                                        actual_unit = 'ml'
                                    elif actual_unit == 'ml' and pred_unit == 'l':
                                        actual_qty /= 1000
                                        actual_unit = 'l'
                                        
                                # Calculate error metrics
                                error = pred_qty - actual_qty
                                error_pct = (error / actual_qty) * 100 if actual_qty != 0 else float('inf')
                                
                                # Store metrics in overall results
                                results['overall']['predictions'].append(pred_qty)
                                results['overall']['actual'].append(actual_qty)
                                results['overall']['errors'].append(error)
                                
                                # Store metrics by meal type
                                if meal_t not in results['by_meal_type']:
                                    results['by_meal_type'][meal_t] = {'predictions': [], 'actual': [], 'errors': []}
                                results['by_meal_type'][meal_t]['predictions'].append(pred_qty)
                                results['by_meal_type'][meal_t]['actual'].append(actual_qty)
                                results['by_meal_type'][meal_t]['errors'].append(error)
                                
                                # Store metrics by event type
                                if evt_type not in results['by_event_type']:
                                    results['by_event_type'][evt_type] = {'predictions': [], 'actual': [], 'errors': []}
                                results['by_event_type'][evt_type]['predictions'].append(pred_qty)
                                results['by_event_type'][evt_type]['actual'].append(actual_qty)
                                results['by_event_type'][evt_type]['errors'].append(error)
                                
                                # Store metrics by category
                                if category not in results['by_category']:
                                    results['by_category'][category] = {'predictions': [], 'actual': [], 'errors': []}
                                results['by_category'][category]['predictions'].append(pred_qty)
                                results['by_category'][category]['actual'].append(actual_qty)
                                results['by_category'][category]['errors'].append(error)
                                
                                # Store item-specific metrics
                                if item not in results['by_item']:
                                    results['by_item'][item] = {
                                        'predictions': [], 'actual': [], 'errors': [], 'category': category
                                    }
                                results['by_item'][item]['predictions'].append(pred_qty)
                                results['by_item'][item]['actual'].append(actual_qty)
                                results['by_item'][item]['errors'].append(error)
                                
                                # Store detailed results for Excel output
                                detailed_results.append({
                                    'Order_Number': order_num,
                                    'Item': item,
                                    'Category': category,
                                    'Predicted_Quantity': pred_qty,
                                    'Predicted_Unit': pred_unit,
                                    'Actual_Quantity': actual_qty,
                                    'Actual_Unit': actual_unit,
                                    'Error': error,
                                    'Error_Percentage': error_pct,
                                    'Meal_Type': meal_t,
                                    'Event_Type': evt_type
                                })
                                
                                prediction_count += 1
                                
                            except Exception as e:
                                skipped_items.append(item)
                                continue
                                
                        except Exception as e:
                            skipped_items.append(item)
                            continue
                    
                    # Store order summary for Excel output
                    order_summary.append({
                        'Order_Number': order_num,
                        'Event_Time': evt_time,
                        'Meal_Time': meal_t,
                        'Event_Type': evt_type,
                        'Total_Guests': guests,
                        'Vegetarian_Guests': vegs,
                        'Total_Items': len(items),
                        'Processed_Items': prediction_count,
                        'Skipped_Items': len(skipped_items)
                    })
                    
                except Exception as e:
                    print(f"Error processing order {order_num}: {str(e)}")
                    continue
                    
                # Update progress counters
                processed_count += 1
                
            # Calculate and display batch statistics
            batch_time = time.time() - batch_start_time
            orders_per_second = (batch_idx * batch_size + len(batch_orders)) / batch_time
            remaining_orders = total_orders - (batch_idx + 1) * batch_size
            eta_seconds = remaining_orders / orders_per_second if orders_per_second > 0 else 0
            
            # Format ETA as mm:ss or hh:mm:ss
            if eta_seconds < 3600:
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                
            print(f"Processed {processed_count}/{total_orders} orders ({processed_count/total_orders:.1%})")
            print(f"Processing speed: {orders_per_second:.2f} orders/sec. ETA: {eta_str}")
            
            # Explicitly trigger garbage collection to free memory
            # This is crucial for Excel processing which tends to consume more memory
            gc.collect()
        
        # Calculate final metrics from results
        metrics = {}
        
        # Calculate overall metrics if we have data
        if results['overall']['actual']:
            overall_actual = np.array(results['overall']['actual'])
            overall_pred = np.array(results['overall']['predictions'])
            
            metrics['overall'] = {
                'MAE': mean_absolute_error(overall_actual, overall_pred),
                'RMSE': np.sqrt(mean_squared_error(overall_actual, overall_pred)),
                'MAPE': np.mean(np.abs((overall_actual - overall_pred) / overall_actual)) * 100,
                'R2': r2_score(overall_actual, overall_pred),
                'Count': len(overall_actual)
            }
        else:
            print("No valid predictions to evaluate.")
            return None
        
        # Calculate metrics by group
        for group_type in ['by_meal_type', 'by_event_type', 'by_category']:
            metrics[group_type] = {}
            for group_name, group_data in results[group_type].items():
                if not group_data['actual']:
                    continue
                    
                group_actual = np.array(group_data['actual'])
                group_pred = np.array(group_data['predictions'])
                
                metrics[group_type][group_name] = {
                    'MAE': mean_absolute_error(group_actual, group_pred),
                    'RMSE': np.sqrt(mean_squared_error(group_actual, group_pred)),
                    'MAPE': np.mean(np.abs((group_actual - group_pred) / group_actual)) * 100,
                    'R2': r2_score(group_actual, group_pred),
                    'Count': len(group_actual)
                }
        
        # Calculate item-level metrics only for items with sufficient data points
        metrics['by_item'] = {}
        for item, item_data in results['by_item'].items():
            if len(item_data['actual']) >= 3:  # Minimum sample size for reliable metrics
                item_actual = np.array(item_data['actual'])
                item_pred = np.array(item_data['predictions'])
                
                metrics['by_item'][item] = {
                    'Category': item_data['category'],
                    'MAE': mean_absolute_error(item_actual, item_pred),
                    'RMSE': np.sqrt(mean_squared_error(item_actual, item_pred)),
                    'MAPE': np.mean(np.abs((item_actual - item_pred) / item_actual)) * 100,
                    'R2': r2_score(item_actual, item_pred),
                    'Count': len(item_actual)
                }
        
        # Convert results to DataFrames and save to Excel
        self._save_metrics_to_excel(metrics, detailed_results, order_summary, output_file)
        
        # Display summary statistics
        print("\n=== Test Evaluation Results ===")
        print(f"MAE: {metrics['overall']['MAE']:.2f}")
        print(f"RMSE: {metrics['overall']['RMSE']:.2f}")
        print(f"MAPE: {metrics['overall']['MAPE']:.2f}%")
        print(f"R²: {metrics['overall']['R2']:.4f}")
        print(f"Total predictions evaluated: {metrics['overall']['Count']}")
        
        # Display per-category performance
        print("\nCategory Performance (MAPE):")
        cat_perf = [(cat, met['MAPE'], met['Count']) 
                    for cat, met in metrics['by_category'].items()]
        cat_perf.sort(key=lambda x: x[1])  # Sort by MAPE (ascending)
        
        # Display top performing categories
        print("Top performing categories:")
        for cat, mape, count in cat_perf[:3]:
            print(f"  {cat}: {mape:.2f}% (n={count})")
            
        # Display categories needing improvement
        print("Categories needing improvement:")
        for cat, mape, count in cat_perf[-3:]:
            print(f"  {cat}: {mape:.2f}% (n={count})")
        
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.1f} seconds")
        print(f"Results saved to {output_file}")
        
        return metrics

    def _initialize_results_structure(self):
        """Initialize nested dictionaries for tracking evaluation metrics"""
        return {
            'overall': {
                'predictions': [],
                'actual': [],
                'errors': []
            },
            'by_meal_type': {},
            'by_event_type': {},
            'by_category': {},
            'by_item': {}
        }

    def _save_metrics_to_excel(self, metrics, detailed_results, order_summary, output_file):
        """Save evaluation metrics to multi-sheet Excel file"""
        # Create DataFrames from results
        
        # Overall metrics
        overall_df = pd.DataFrame([{
            'Metric': 'Overall',
            'MAE': metrics['overall']['MAE'],
            'RMSE': metrics['overall']['RMSE'],
            'MAPE': metrics['overall']['MAPE'],
            'R2': metrics['overall']['R2'],
            'Count': metrics['overall']['Count']
        }])
        
        # Meal type metrics
        meal_rows = []
        for meal_type, meal_metrics in metrics['by_meal_type'].items():
            meal_rows.append({
                'Meal_Type': meal_type,
                'MAE': meal_metrics['MAE'],
                'RMSE': meal_metrics['RMSE'],
                'MAPE': meal_metrics['MAPE'],
                'R2': meal_metrics['R2'],
                'Count': meal_metrics['Count']
            })
        meal_df = pd.DataFrame(meal_rows)
        
        # Event type metrics
        event_rows = []
        for event_type, event_metrics in metrics['by_event_type'].items():
            event_rows.append({
                'Event_Type': event_type,
                'MAE': event_metrics['MAE'],
                'RMSE': event_metrics['RMSE'],
                'MAPE': event_metrics['MAPE'],
                'R2': event_metrics['R2'],
                'Count': event_metrics['Count']
            })
        event_df = pd.DataFrame(event_rows)
        
        # Category metrics
        category_rows = []
        for category, cat_metrics in metrics['by_category'].items():
            category_rows.append({
                'Category': category,
                'MAE': cat_metrics['MAE'],
                'RMSE': cat_metrics['RMSE'],
                'MAPE': cat_metrics['MAPE'],
                'R2': cat_metrics['R2'],
                'Count': cat_metrics['Count']
            })
        category_df = pd.DataFrame(category_rows)
        
        # Item metrics
        item_rows = []
        for item, item_metrics in metrics['by_item'].items():
            item_rows.append({
                'Item': item,
                'Category': item_metrics['Category'],
                'MAE': item_metrics['MAE'],
                'RMSE': item_metrics['RMSE'],
                'MAPE': item_metrics['MAPE'],
                'R2': item_metrics['R2'],
                'Count': item_metrics['Count']
            })
        item_df = pd.DataFrame(item_rows)
        
        # Detailed results and order summary
        detailed_df = pd.DataFrame(detailed_results)
        summary_df = pd.DataFrame(order_summary)
        
        # Write all to Excel
        with pd.ExcelWriter(output_file) as writer:
            overall_df.to_excel(writer, sheet_name='Overall_Metrics', index=False)
            meal_df.to_excel(writer, sheet_name='Meal_Type_Metrics', index=False)
            event_df.to_excel(writer, sheet_name='Event_Type_Metrics', index=False)
            category_df.to_excel(writer, sheet_name='Category_Metrics', index=False)
            item_df.sort_values(by='MAPE').to_excel(writer, sheet_name='Item_Metrics', index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Order_Summary', index=False)