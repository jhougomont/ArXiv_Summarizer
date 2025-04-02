import requests
import feedparser
import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
import time # Import time for parsing dates
import sys # Import sys for exiting
import re # Import regular expression module
import argparse # Import argument parsing module

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
# Expects OPENAI_API_KEY environment variable to be set
try:
    client = OpenAI()
    openai_enabled = True
except Exception as e:
    print(f"Warning: OpenAI client initialization failed. Summarization disabled. Error: {e}")
    openai_enabled = False

OUTPUT_DIR = "output" # Directory to save markdown files
FETCH_BATCH_SIZE = 500 # Fetch more papers initially to ensure we get all of yesterday's

def fetch_arxiv_papers(categories, max_results=FETCH_BATCH_SIZE):
    """Fetches recent papers from arXiv API for given categories."""
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = '+OR+'.join([f'cat:{cat}' for cat in categories])
    query = f'search_query={search_query}&sortBy=lastUpdatedDate&sortOrder=descending&max_results={max_results}'
    
    # print(f"Fetching up to {max_results} recent papers from arXiv...")
    # print(f"API URL: {base_url + query}")
    
    try:
        response = requests.get(base_url + query)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during arXiv API request: {e}")
        return []
        
    feed = feedparser.parse(response.content)
    # print(f"Received {len(feed.entries)} entries from API.")
    return feed.entries

def summarize_text(text, model="gpt-3.5-turbo"):
    """Summarizes text using OpenAI API."""
    if not openai_enabled:
        return "Summarization disabled (OpenAI client not initialized)."
    if not text or text.isspace():
        return "(Abstract empty or missing)"
        
    try:
        # Add a small delay to avoid hitting rate limits too quickly
        # time.sleep(0.5)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic paper abstracts concisely in 1-2 sentences."},
                {"role": "user", "content": f"""Please summarize the following abstract:

{text}"""}
            ],
            temperature=0.5,
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"\nError during OpenAI summarization: {e}")
        return "(Could not summarize due to API error)"

def summarize_category(category_name, numbered_summaries_text, model="gpt-3.5-turbo"):
    """Summarizes a collection of numbered summaries for a specific category using OpenAI API, requesting citations."""
    if not openai_enabled:
        return "Summarization disabled (OpenAI client not initialized)."
    if not numbered_summaries_text or numbered_summaries_text.isspace():
        return "(No summaries provided for category)"
        
    prompt = f"""Please provide a 2-3 sentence summary highlighting the main topics or trends from the following paper summaries in the {category_name} category released yesterday. 
IMPORTANT: When referring to points from specific papers, cite the paper number provided in brackets, like `[1]`, `[5]` etc.

{numbered_summaries_text}"""
    
    try:
        # time.sleep(0.5) # Optional delay
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that synthesizes key themes from a list of research paper summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=200, # Allow slightly longer summary for category
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"\nError during OpenAI category summarization for {category_name}: {e}")
        return "(Could not summarize category due to API error)"

def main():
    parser = argparse.ArgumentParser(description="Fetch and summarize arXiv papers from the previous day.")
    parser.add_argument(
        '--keywords', 
        nargs='+', 
        help='One or more keywords to prioritize (case-insensitive). Papers matching these keywords in title/abstract will be listed first.',
        default=[]
    )
    parser.add_argument(
        '--ocean', 
        action='store_true', 
        help='Use a predefined list of ocean science keywords for prioritization.'
    )
    parser.add_argument(
        '--categories', 
        nargs='+', 
        help='Specify which arXiv categories to fetch (e.g., cs.AI cs.LG). Defaults to cs.AI, cs.CV, cs.CY, cs.LG.',
        default=['cs.AI', 'cs.CV', 'cs.CY', 'cs.LG']
    )
    args = parser.parse_args()

    categories_of_interest = args.categories
    
    # --- Determine Keywords for Prioritization ---
    keywords_to_prioritize = []
    if args.ocean:
        print("Prioritizing using predefined Ocean Science keywords.")
        keywords_to_prioritize = [
            'ocean', 'marine', 'sea', 'coastal', 'maritime', 'offshore', 
            'hydrography', 'oceanography', 'bathymetry', 'aquatic'
        ]
    elif args.keywords:
        print(f"Prioritizing using custom keywords: {args.keywords}")
        keywords_to_prioritize = args.keywords
    else:
        print("No keyword prioritization specified.")
        
    prioritize_keywords_lower = [k.lower() for k in keywords_to_prioritize]
    use_keyword_prioritization = bool(prioritize_keywords_lower)
    # ------------------------------------------

    # --- Date Calculation --- 
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    yesterday_utc = now_utc - datetime.timedelta(days=1)
    yesterday_date_str = yesterday_utc.strftime("%Y-%m-%d")
    print(f"Targeting papers from yesterday (UTC): {yesterday_date_str}")
    # ------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.join(OUTPUT_DIR, f"arxiv_summary_{yesterday_date_str}.md")
    
    print(f"Fetching recent papers for categories: {', '.join(categories_of_interest)}...")
    
    all_recent_papers = fetch_arxiv_papers(categories_of_interest, max_results=FETCH_BATCH_SIZE)
    
    if not all_recent_papers:
        print("Could not fetch any papers from arXiv. Exiting.")
        # Optionally write an empty file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# arXiv Summary for {yesterday_date_str}\n\n")
            f.write("Failed to fetch papers from arXiv.\n")
        return
        
    # --- Filter for Yesterday's Papers --- 
    yesterdays_papers = []
    print(f"Filtering {len(all_recent_papers)} fetched papers for date {yesterday_date_str} and primary category match...")
    for entry in all_recent_papers:
        try:
            # Use parsed dates which are structs in UTC
            published_dt = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
            updated_dt = datetime.datetime(*entry.updated_parsed[:6], tzinfo=datetime.timezone.utc) if hasattr(entry, 'updated_parsed') else published_dt
            
            # Check primary category first
            primary_category = entry.get('arxiv_primary_category', {}).get('term')
            if not primary_category or primary_category not in categories_of_interest:
                continue # Skip if primary category doesn't match interests

        except Exception as e:
            print(f"Warning: Could not parse date or category for entry {entry.get('id', 'N/A')}: {e}")
            continue # Skip if basic info is missing/unparseable
            
        target_date = yesterday_utc.date()
        published_date = published_dt.date()
        updated_date = updated_dt.date()

        # Now check date: Include if it was *updated* yesterday OR *published* yesterday.
        if updated_date == target_date or published_date == target_date:
             yesterdays_papers.append(entry)
             
    # -------------------------------------

    num_found = len(yesterdays_papers)
    print(f"\nFound {num_found} papers published or updated on {yesterday_date_str}.")

    if num_found == 0:
        print("No relevant papers found for yesterday.")
        # Write message to file and exit
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# arXiv Summary for {yesterday_date_str}\n\n")
            f.write(f"Categories: `{', '.join(categories_of_interest)}`\n")
            f.write("Papers found: 0\n")
            f.write("--------------------\n\n")
            f.write("No new papers found for the specified categories published or updated on this date.\n")
        print(f"Output written to {output_filename}")
        return
        
    # --- Interactive Prompt --- 
    papers_to_process = []
    while True:
        choice = input(f"Proceed with all {num_found} papers, or specify a number per category? (all / number / quit): ").lower().strip()
        
        if choice == 'all':
            papers_to_process = yesterdays_papers
            print(f"Proceeding with all {len(papers_to_process)} papers.")
            break
        elif choice == 'quit':
            print("Exiting without processing.")
            sys.exit()
        elif choice == 'number':
            try:
                limit_per_category = int(input("Enter the maximum number of papers per category to summarize: "))
                if limit_per_category <= 0:
                    print("Please enter a positive number.")
                    continue
                    
                # Categorize and limit
                papers_by_category = {cat: [] for cat in categories_of_interest}
                skipped_papers_count = 0

                for paper in yesterdays_papers:
                    assigned_category = None
                    if hasattr(paper, 'tags'):
                        for tag in paper.tags:
                            if tag.term in categories_of_interest:
                                assigned_category = tag.term
                                break # Assign to the first matching category of interest
                    
                    if assigned_category:
                        # Papers are already sorted by date desc by API, so just append
                        papers_by_category[assigned_category].append(paper)
                    else:
                        skipped_papers_count += 1
                
                if skipped_papers_count > 0:
                     print(f"Note: Skipped {skipped_papers_count} papers that didn't list a primary category matching your interests.")

                temp_papers_list = []
                for category, papers_in_cat in papers_by_category.items():
                    # Take the top N (which are the most recent due to API sort)
                    limited_list = papers_in_cat[:limit_per_category]
                    temp_papers_list.extend(limited_list)
                    # print(f"  Selected {len(limited_list)} papers for category {category}") # Debug
                
                papers_to_process = temp_papers_list
                # Optional: Re-sort the final list by date if needed, currently grouped by category
                # papers_to_process.sort(key=lambda p: p.updated_parsed, reverse=True)
                
                print(f"Selected {len(papers_to_process)} papers based on a limit of {limit_per_category} per category.")
                break
                
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                 print(f"An error occurred during selection: {e}")
                 # Offer to proceed with 'all' or 'quit' as fallback?
                 continue # Re-prompt the user
        else:
            print("Invalid choice. Please enter 'all', 'number', or 'quit'.")
    # ------------------------

    num_to_process = len(papers_to_process)
    if num_to_process == 0:
        print("No papers selected for processing.")
        # Write message to file and exit
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# arXiv Summary for {yesterday_date_str}\n\n")
            f.write(f"Categories: `{', '.join(categories_of_interest)}`\n")
            f.write("Papers found (yesterday): {num_found}\n")
            f.write("Papers selected for processing: 0\n")
            f.write("--------------------\n\n")
            f.write("No papers selected based on user input.\n")
        print(f"Output written to {output_filename}")
        return
        
    print(f"\nStarting summarization for {num_to_process} selected papers...")

    # --- Define Ocean Keywords --- 
    OCEAN_KEYWORDS = [
        'ocean', 'marine', 'sea', 'coastal', 'maritime', 'offshore', 
        'hydrography', 'oceanography', 'bathymetry', 'aquatic'
    ]
    ocean_keywords_lower = [k.lower() for k in OCEAN_KEYWORDS]
    # ---------------------------

    # --- Step 1: Process all selected papers (summarize abstracts) --- 
    final_processed_papers = []
    papers_by_category_final = {cat: [] for cat in categories_of_interest} # For category summaries
    
    for i, entry in enumerate(papers_to_process):
        print(f"Processing paper {i+1}/{num_to_process}...") # Simple progress indicator
        # Find the primary category 
        primary_cat = "N/A"
        if hasattr(entry, 'tags'):
             for tag in entry.tags:
                 if tag.term in categories_of_interest:
                     primary_cat = tag.term
                     break
        entry.primary_category = primary_cat # Store for later use

        # Summarize the abstract 
        title_lower = entry.title.lower()
        abstract = entry.summary.replace('\n', ' ').strip()
        abstract_lower = abstract.lower()
        print(f"  Summarizing abstract for: [{primary_cat}] {title_lower}")
        summary = summarize_text(abstract)
        if "(Could not summarize" in summary:
             print(f"  Warning: Failed to summarize paper {i+1}")
        entry.generated_summary = summary 
        
        # Check for prioritization keywords if enabled
        entry.is_prioritized = False
        if use_keyword_prioritization:
            for keyword in prioritize_keywords_lower:
                # Use regex word boundaries to match whole words only
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title_lower) or re.search(pattern, abstract_lower):
                    entry.is_prioritized = True
                    print(f"    -> Flagged for prioritization (keyword: {keyword})")
                    break
        
        # Assign final number (important: add 1 because list is 0-indexed)
        # This number might change after sorting
        entry.final_number = i + 1
        
        final_processed_papers.append(entry)
        
        # Group for initial category summary generation
        if primary_cat in papers_by_category_final:
             papers_by_category_final[primary_cat].append(entry)
    
    print(f"Finished processing {len(final_processed_papers)} papers.")

    # --- Step 2: Generate Category Summaries (with citation instructions) --- 
    print("\nGenerating category summaries...")
    category_summaries_output = {}
    
    for category, papers_in_cat in papers_by_category_final.items():
        if not papers_in_cat:
            continue 
        
        print(f"  Preparing summary for category: {category} ({len(papers_in_cat)} papers)")
        
        # Collect successful individual summaries with their final numbers
        summaries_to_summarize = [
            (p.final_number, p.generated_summary) 
            for p in papers_in_cat 
            if hasattr(p, 'generated_summary') and p.generated_summary and "Could not summarize" not in p.generated_summary
        ]
        
        if not summaries_to_summarize:
             category_summaries_output[category] = "(No successful individual summaries available to generate category summary)"
             print(f"    Skipping {category} summary (no individual summaries).")
        else:
             # Limit the number of summaries to avoid excessive prompt length/cost
             max_summaries_for_cat_summary = 15 
             if len(summaries_to_summarize) > max_summaries_for_cat_summary:
                  print(f"    Note: Using first {max_summaries_for_cat_summary} summaries for {category} due to limit.")
             summaries_to_summarize = summaries_to_summarize[:max_summaries_for_cat_summary]
             
             # Format with numbers for the prompt
             numbered_summaries_text_block = "\n\n".join(f"[{num}] {s}" for num, s in summaries_to_summarize)
             category_summary_text = summarize_category(category, numbered_summaries_text_block) 
             category_summaries_output[category] = category_summary_text
                 
    print("Category summaries generated.")

    # --- Step 2.5: Sort final papers to prioritize based on keywords (if enabled) --- 
    category_summaries_to_write = category_summaries_output # Default to initially generated ones
    
    if use_keyword_prioritization:
        print("Sorting papers to prioritize keyword matches...")
        # Sort key: Puts is_prioritized=True first, then uses original number
        final_processed_papers.sort(key=lambda p: (not p.is_prioritized, p.final_number))
        # Update final numbers after sorting
        for new_idx, entry in enumerate(final_processed_papers):
            entry.final_number = new_idx + 1
            
        # Rebuild the input for category summarization with the *new* numbers
        papers_by_category_final_sorted = {cat: [] for cat in categories_of_interest}
        for entry in final_processed_papers:
            if entry.primary_category in papers_by_category_final_sorted:
                 papers_by_category_final_sorted[entry.primary_category].append(entry)
        
        print("Re-generating category summaries with updated numbering for citations...")
        category_summaries_output_sorted = {} # Re-initialize
        for category, papers_in_cat in papers_by_category_final_sorted.items():
            if not papers_in_cat:
                continue 
            summaries_to_summarize = [
                (p.final_number, p.generated_summary) 
                for p in papers_in_cat 
                if hasattr(p, 'generated_summary') and p.generated_summary and "Could not summarize" not in p.generated_summary
            ]
            if not summaries_to_summarize:
                 category_summaries_output_sorted[category] = "(No successful individual summaries available to generate category summary)"
            else:
                 max_summaries_for_cat_summary = 15 
                 summaries_to_summarize = summaries_to_summarize[:max_summaries_for_cat_summary]
                 numbered_summaries_text_block = "\n\n".join(f"[{num}] {s}" for num, s in summaries_to_summarize)
                 category_summary_text = summarize_category(category, numbered_summaries_text_block) 
                 category_summaries_output_sorted[category] = category_summary_text
        category_summaries_to_write = category_summaries_output_sorted # Use the re-generated ones
    # --------------------------------------------------------------------------

    # --- Step 3: Write Markdown File --- 
    print(f"\nWriting output to {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        # --- Header --- 
        f.write(f"# arXiv Summary for {yesterday_date_str}\n\n")
        f.write(f"Categories: `{', '.join(categories_of_interest)}`\n")
        f.write(f"Papers Found (published/updated yesterday): {num_found}\n")
        f.write(f"Papers Processed: {num_to_process}\n")
        f.write("--------------------\n\n")

        # --- Category Summaries (Now at the top, using potentially sorted results) --- 
        f.write("# Category Summaries\n\n")
        if not category_summaries_to_write:
             f.write("(No category summaries generated)\n\n")
        else:
             sorted_categories = sorted(category_summaries_to_write.keys())
             
             for category in sorted_categories:
                 if category in categories_of_interest: # Only print summaries for requested categories
                     summary_text = category_summaries_to_write[category]
                     f.write(f"## {category}\n\n")
                     f.write(f"{summary_text}\n\n")
                     f.write("--------------------\n\n")
        
        # --- Detailed Paper List (Now sorted with highlights) --- 
        f.write("# Paper Details\n\n")
        if not final_processed_papers:
            f.write("(No papers were processed)\n")
        else:
            for entry in final_processed_papers:
                # Retrieve data stored on the entry object
                i = entry.final_number
                title = entry.title.replace('\n', ' ').strip()
                authors = ', '.join(author.name for author in entry.authors)
                published_display_date = entry.published.split('T')[0]
                updated_display_date = entry.updated.split('T')[0] if hasattr(entry, 'updated') else published_display_date
                primary_cat = entry.primary_category
                display_date_note = f" (Updated: {updated_display_date})" if updated_display_date != published_display_date else ""
                link = entry.link
                pdf_link = link.replace('/abs/', '/pdf/') + '.pdf'
                summary = entry.generated_summary

                # Write entry to markdown file
                highlight_prefix = "" 
                if use_keyword_prioritization and entry.is_prioritized:
                    highlight_prefix = "**[KEYWORD MATCH]** "
                f.write(f"## {i}. {highlight_prefix}{title}\n\n") # Use final_number here
                f.write(f"*   **Category:** `{primary_cat}`\n")
                f.write(f"*   **Authors:** {authors}\n")
                f.write(f"*   **Published:** {published_display_date}{display_date_note}\n")
                f.write(f"*   **Link:** [{link}]({link})\n")
                f.write(f"*   **PDF Link:** [{pdf_link}]({pdf_link})\n")
                f.write(f"*   **Summary:** {summary}\n\n")
                f.write("--------------------\n\n")

    print(f"Output successfully written to {output_filename}")

if __name__ == "__main__":
    main() 