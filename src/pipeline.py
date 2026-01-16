from transformers import pipeline
import pandas as pd
import torch

def create_llm_table_generator():
    """
    Create a simple LLM pipeline for table generation
    """
    print("Loading Llama-3.1-8B-Instruct model...")
    
    # Use a smaller, more accessible model for demonstration
    # You can change this to "meta-llama/Llama-3.1-8B-Instruct" when you have access
    model_name = "meta-llama/Llama-3.1-8B-Instruct" # Replace with actual Llama model
    
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=False
    )
    
    return generator

def generate_table_from_csv(csv_path, output_format="markdown"):
    """
    Generate a formatted table from CSV using LLM
    
    Args:
        csv_path (str): Path to CSV file
        output_format (str): "markdown" or "html"
    
    Returns:
        str: Formatted table
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Convert to text representation
    csv_text = df.to_csv(index=False)
    
    # Create prompt for LLM
    prompt = f"""Convert this CSV data into a clean {output_format} table:

            CSV Data:
            {csv_text}

            Generate a properly formatted {output_format} table with headers and data. Make it readable and well-structured.

            {output_format.upper()} Table:"""
    
    # Initialize generator
    generator = create_llm_table_generator()
    
    # Generate table
    print(f"Generating {output_format} table from {csv_path}...")
    result = generator(prompt)[0]['generated_text']
    
    # Extract just the table part (remove the prompt)
    table_output = result.replace(prompt, "").strip()
    
    return table_output

def process_csv_to_table(csv_path, output_file=None):
    """
    Process a CSV file and generate both markdown and HTML tables
    
    Args:
        csv_path (str): Input CSV file path
        output_file (str): Optional output file path (without extension)
    """
    try:
        # Generate HTML table
        html_table = generate_table_from_csv(csv_path, "html")
        
        print("\n=== HTML TABLE ===")
        print(html_table)
        
        # Save to files if output_file specified
        if output_file:
            with open(f"{output_file}.html", "w") as f:
                f.write(f"<html><body><h1>Table from {csv_path}</h1>{html_table}</body></html>")
            
            print(f"\nTables saved to {output_file}.md and {output_file}.html")
        
        return {
            "html": html_table
        }
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

# Simple usage
if __name__ == "__main__":
    # Process a specific CSV file
    csv_file = "extracted_tables/page_1_table_1.csv"  # Change this to your CSV file
    
    if pd.io.common.file_exists(csv_file):
        result = process_csv_to_table(csv_file, "generated_table")
    else:
        print(f"CSV file not found: {csv_file}")
        print("Available CSV files:")
        import os
        for file in os.listdir("extracted_tables"):
            if file.endswith(".csv"):
                print(f"  - {file}")