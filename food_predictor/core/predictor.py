import pandas as pd
import numpy as np
import logging
import argparse

from food_predictor.core.model_manager import ModelManager
from food_predictor.core.feature_engineering import FeatureEngineer
from food_predictor.core.category_rules import FoodCategoryRules
from food_predictor.data.item_service import ItemService
from food_predictor.core.menu_analyzer import MenuAnalyzer
from food_predictor.pricing.price_calculator import PriceCalculator
from food_predictor.utils.quantity_utils import extract_quantity_value

def main():
    parser = argparse.ArgumentParser(description="Food Quantity Predictor & Price Calculator")
    parser.add_argument('--data', type=str, default="DB38.xlsx", help="Path to training data")
    parser.add_argument('--mode', choices=['train', 'predict', 'price', 'evaluate'], default='train')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--eval_output', type=str, default='evaluation_results.xlsx', 
                        help="Path to save evaluation results (only used in 'evaluate' mode)")
    parser.add_argument('--num_orders', type=int, default=100, 
                        help="Number of orders to randomly evaluate (only used in 'evaluate' mode)")
    parser.add_argument('--eval_type', choices=['random', 'test_set'], default='random',
                        help="Evaluation type: random samples or external test set")
    parser.add_argument('--test_data', type=str, default=None,
                        help="Path to external test dataset (only used with --eval_type=test_set)")
    parser.add_argument('--batch_size', type=int, default=50,
                        help="Batch size for test set evaluation (memory optimization)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Initialize core components
    food_rules = FoodCategoryRules()
    item_service = ItemService(food_rules)
    feature_engineer = FeatureEngineer(food_rules)
    menu_analyzer = MenuAnalyzer(food_rules, item_service)
    model_manager = ModelManager(feature_engineer, item_service, food_rules,menu_analyzer)
    price_calculator = PriceCalculator(food_rules, item_service)

    if args.mode == 'train':
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        model_manager.fit(data)
        model_manager.save_models(args.model_dir)
        print("Training complete and model saved.")

    elif args.mode == 'predict':
        data = pd.read_excel(args.data)
        item_service.build_item_metadata(data)
        model_manager.load_models(args.model_dir)
        # Example: get prediction for a sample menu
        event_time = "Evening"
        meal_type = "Dinner"
        event_type = "Wedding"
        total_guest_count = 100
        veg_guest_count = 40
        selected_items = ["Paneer Butter Masala", "Biryani", "Curd", "Papad"]
        menu_analyzer.build_menu_context(event_time,meal_type,event_type,total_guest_count,veg_guest_count,selected_items)
        

        predictions = model_manager.predict(
            event_time, meal_type, event_type, total_guest_count, veg_guest_count,selected_items
        )
        if args.mode == "predict":
         for item, pred in predictions.items():
            print(f"{item}: {pred['total']} ({pred['unit']})")

    elif args.mode == 'price':
        model_manager.load_models(args.model_dir)
        # Example: get prediction and price for a sample menu
        event_time = "Evening"
        meal_type = "Dinner"
        event_type = "Wedding"
        total_guest_count = 100
        veg_guest_count = 40
        selected_items = ["Paneer Butter Masala", "Biryani", "Curd", "Papad"]

        predictions = model_manager.predict(
            event_time, meal_type, event_type, total_guest_count, veg_guest_count,selected_items
        )
        for item, pred in predictions.items():
            qty = extract_quantity_value(pred['total'])
            unit = pred['unit']
            category = pred['category']
            total_price, base_price_per_unit, per_person_price = price_calculator.calculate_price(
                converted_qty=qty,
                category=category,
                total_guest_count=total_guest_count,
                item_name=item,
                unit=unit
            )
            print(f"{item}: {pred['total']} ({unit}) | Total Price: ₹{total_price:.2f} | Per Person: ₹{per_person_price:.2f}")
    
    elif args.mode == 'evaluate':
    # Load models for either evaluation type
        model_manager.load_models(args.model_dir)
    
        if args.eval_type == 'random':
            print(f"\n=== Evaluating Model on Random {args.num_orders} Orders ===\n")
            
            # Load data for random evaluation
            data = pd.read_excel(args.data)
            item_service.build_item_metadata(data)
            
            # Use the random evaluation method
            evaluation_results = model_manager.evaluate_item_accuracy(
                data=data,
                food_rules=food_rules,
                feature_engineer=feature_engineer,
                item_matcher=item_service.item_matcher,
                num_orders=args.num_orders,
                random_state=42,
                output_file=args.eval_output
            )
            
            print(f"\nRandom evaluation complete! Results saved to {args.eval_output}")
            
        elif args.eval_type == 'test_set':
            if not args.test_data:
                print("Error: --test_data parameter is required for test_set evaluation")
                return
                
            print(f"\n=== Evaluating Model on External Test Dataset ===\n")
            print(f"Test data path: {args.test_data}")
            
            # Build item metadata from the main dataset
            data = pd.read_excel(args.data)
            item_service.build_item_metadata(data)
            
            # Use the external test set evaluation method
            evaluation_results = model_manager.evaluate_on_external_test_set(
                test_data_path=args.test_data,
                output_file=args.eval_output,
                batch_size=args.batch_size
            )
            
            print(f"\nExternal test set evaluation complete! Results saved to {args.eval_output}")
if __name__ == "__main__":
    main()
