#!/usr/bin/env python3
"""
Script to update the disease remedies dictionary in the plant_disease_detection.py file.
This allows adding new disease information and remedies without modifying the main script.
"""

import os
import json
import re
import argparse

def load_current_remedies():
    """Load the current DISEASE_REMEDIES dictionary from the main script"""
    try:
        # Try to extract the dictionary from the main file
        with open('plant_disease_detection.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the dictionary in the file
        pattern = r'DISEASE_REMEDIES\s*=\s*\{(.*?)\}\s*\n\s*# Default remedy'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("Could not find DISEASE_REMEDIES dictionary in plant_disease_detection.py")
            return {}
        
        # Extract the dictionary string and evaluate it
        dict_str = '{' + match.group(1) + '}'
        
        # Replace single quotes with double quotes for JSON parsing
        dict_str = dict_str.replace("'", '"')
        
        # Clean up the string to make it valid JSON
        dict_str = re.sub(r'#.*?\n', '\n', dict_str)  # Remove comments
        dict_str = re.sub(r',(\s*})', r'\1', dict_str)  # Remove trailing commas
        
        # Parse the JSON
        remedies = json.loads(dict_str)
        return remedies
    
    except Exception as e:
        print(f"Error loading current remedies: {e}")
        return {}

def update_remedies_in_file(remedies):
    """Update the DISEASE_REMEDIES dictionary in the main script"""
    try:
        with open('plant_disease_detection.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the dictionary in the file
        pattern = r'(DISEASE_REMEDIES\s*=\s*\{)(.*?)(\}\s*\n\s*# Default remedy)'
        
        # Format the new dictionary
        formatted_dict = "DISEASE_REMEDIES = {\n"
        for disease, info in remedies.items():
            formatted_dict += f'    "{disease}": {{\n'
            formatted_dict += f'        "description": "{info["description"]}",\n'
            if "causes" in info:
                formatted_dict += '        "causes": [\n'
                for cause in info["causes"]:
                    formatted_dict += f'            "{cause}",\n'
                formatted_dict += '        ],\n'
            if "symptoms" in info:
                formatted_dict += '        "symptoms": [\n'
                for symptom in info["symptoms"]:
                    formatted_dict += f'            "{symptom}",\n'
                formatted_dict += '        ],\n'
            formatted_dict += '        "remedies": [\n'
            for remedy in info["remedies"]:
                formatted_dict += f'            "{remedy}",\n'
            formatted_dict += '        ],\n'
            if "maintenance" in info:
                formatted_dict += '        "maintenance": [\n'
                for item in info["maintenance"]:
                    formatted_dict += f'            "{item}",\n'
                formatted_dict += '        ],\n'
            if "severity" in info:
                formatted_dict += f'        "severity": "{info["severity"]}"\n'
            formatted_dict += '    },\n'
        formatted_dict += '}'
        
        # Replace the old dictionary with the new one
        updated_content = re.sub(pattern, r'\1' + formatted_dict[18:] + r'\3', content, flags=re.DOTALL)
        
        # Write the updated content back to the file
        with open('plant_disease_detection.py', 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("Successfully updated DISEASE_REMEDIES in plant_disease_detection.py")
        
    except Exception as e:
        print(f"Error updating remedies in file: {e}")

def add_disease(remedies, disease_name, description, remedies_list):
    """Add a new disease to the remedies dictionary"""
    if disease_name in remedies:
        print(f"Warning: Disease '{disease_name}' already exists. Updating information.")
    
    remedies[disease_name] = {
        "description": description,
        "remedies": remedies_list
    }
    
    return remedies

def export_remedies_to_json(remedies, output_file):
    """Export the remedies dictionary to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(remedies, f, indent=4, ensure_ascii=False)
        print(f"Successfully exported remedies to {output_file}")
    except Exception as e:
        print(f"Error exporting remedies to JSON: {e}")

def import_remedies_from_json(input_file):
    """Import remedies from a JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            remedies = json.load(f)
        print(f"Successfully imported remedies from {input_file}")
        return remedies
    except Exception as e:
        print(f"Error importing remedies from JSON: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Update disease remedies dictionary')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Add disease command
    add_parser = subparsers.add_parser('add', help='Add a new disease')
    add_parser.add_argument('--name', type=str, required=True, 
                           help='Disease name (e.g., "Tomato___Leaf_Mold")')
    add_parser.add_argument('--description', type=str, required=True,
                           help='Disease description')
    add_parser.add_argument('--remedies', type=str, nargs='+', required=True,
                           help='List of remedies')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export remedies to JSON')
    export_parser.add_argument('--output', type=str, default='disease_remedies.json',
                              help='Output JSON file')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import remedies from JSON')
    import_parser.add_argument('--input', type=str, required=True,
                              help='Input JSON file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all diseases')
    
    args = parser.parse_args()
    
    # Load current remedies
    remedies = load_current_remedies()
    
    if args.command == 'add':
        # Add new disease
        remedies = add_disease(remedies, args.name, args.description, args.remedies)
        update_remedies_in_file(remedies)
        
    elif args.command == 'export':
        # Export remedies to JSON
        export_remedies_to_json(remedies, args.output)
        
    elif args.command == 'import':
        # Import remedies from JSON
        imported_remedies = import_remedies_from_json(args.input)
        if imported_remedies:
            update_remedies_in_file(imported_remedies)
            
    elif args.command == 'list':
        # List all diseases
        print("\nAvailable diseases and remedies:")
        for disease, info in remedies.items():
            print(f"\n{disease.replace('___', ' - ').replace('_', ' ')}:")
            print(f"  Description: {info['description']}")
            if "causes" in info:
                print("  Causes:")
                for cause in info["causes"]:
                    print(f"    - {cause}")
            if "symptoms" in info:
                print("  Symptoms:")
                for symptom in info["symptoms"]:
                    print(f"    - {symptom}")
            print("  Remedies:")
            for remedy in info["remedies"]:
                print(f"    - {remedy}")
            if "maintenance" in info:
                print("  Maintenance:")
                for item in info["maintenance"]:
                    print(f"    - {item}")
            if "severity" in info:
                print(f"  Severity: {info['severity']}")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 