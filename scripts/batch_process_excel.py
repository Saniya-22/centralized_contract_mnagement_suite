import asyncio
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import GovGigOrchestrator
from src.config import settings


async def process_batch(input_file: str, output_file: str):
    """
    Reads an Excel file, processes each 'User Query' through the GovGigOrchestrator,
    and saves the results to a new column 'Latest Responses'.
    """
    print(f"[{datetime.now()}] Starting batch processing for {input_file}...")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load Excel
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    if "User Query" not in df.columns:
        print(f"Error: 'User Query' column not found in {input_file}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Initialize Orchestrator
    orchestrator = GovGigOrchestrator()

    results = []
    total = len(df)

    print(
        f"[{datetime.now()}] Found {total} queries to process. Using model: {settings.SYNTHESIZER_MODEL}"
    )

    for index, row in df.iterrows():
        query = str(row["User Query"]).strip()
        if not query or query == "nan":
            print(f"[{index+1}/{total}] Skipping empty query.")
            results.append("Empty Query")
            continue

        print(f"[{index+1}/{total}] Processing: {query[:50]}...")

        try:
            # Create a unique thread_id for each query to avoid state mixing
            thread_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}"

            # Run the orchestrator
            response = await orchestrator.run_async(
                query=query,
                context={
                    "thread_id": thread_id,
                    "user_id": "batch_worker"
                }
            )

            # Extract text response
            output_text = response.get("response", "No response generated")
            results.append(output_text)

        except Exception as e:
            print(f"Error processing row {index+1}: {e}")
            results.append(f"Error: {str(e)}")

    # Add new column
    df["latest responses"] = results

    # Save output
    try:
        df.to_excel(output_file, index=False)
        print(f"[{datetime.now()}] Success! Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    INPUT_FILE = "new_responses.xlsx"
    OUTPUT_FILE = "new_responses_updated.xlsx"

    asyncio.run(process_batch(INPUT_FILE, OUTPUT_FILE))
