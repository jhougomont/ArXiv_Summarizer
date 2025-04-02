# arXiv Paper Summarizer

This script fetches and summarizes arXiv papers from specific computer science categories published or updated the previous day, with special prioritization for ocean science content.

## Features

-   Fetches papers from specified arXiv categories (`cs.AI`, `cs.CV`, `cs.CY`, `cs.LG` by default).
-   Filters papers to include only those primarily categorized under the specified fields and published/updated on the previous day (UTC).
-   Uses the OpenAI API (`gpt-3.5-turbo`) to summarize individual paper abstracts.
-   **Prioritizes Ocean Science:**
    -   Identifies papers mentioning keywords related to ocean science (e.g., 'ocean', 'marine', 'coastal') in their title or abstract using whole-word matching.
    -   Sorts the final list to place these relevant papers at the top.
    -   Highlights ocean-related papers with `**[OCEAN RELATED]**` in the output.
-   Generates a 2-3 sentence summary for each category based on the summaries of papers within it, attempting to cite the relevant paper numbers (`[1]`, `[2]`, etc.).
-   **Interactive Selection:** Prompts the user to choose whether to process all found papers or limit the number per category.
-   Saves the output to a dated Markdown file in the `output/` directory, with category summaries at the top followed by the prioritized, detailed paper list.

## Categories Fetched

-   Artificial Intelligence (`cs.AI`)
-   Computer Vision and Pattern Recognition (`cs.CV`)
-   Computers and Society (`cs.CY`)
-   Machine Learning (`cs.LG`)

## Setup

1.  **Clone the repository (or create the files):**
    Ensure you have `arxiv_summarizer.py`, `requirements.txt`, and a `.env` file in the same directory.

2.  **Create `.env` file:**
    This file must contain your OpenAI API key:
    ```
    OPENAI_API_KEY='your_api_key_here'
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Run the script from your terminal. You can use command-line options to customize its behavior:
    ```bash
    python arxiv_summarizer.py [--categories CAT1 CAT2...] [--keywords KW1 KW2...] [--ocean]
    ```
    *   `--categories CAT1 CAT2...`: Specify arXiv categories (default: `cs.AI cs.CV cs.CY cs.LG`).
    *   `--keywords KW1 KW2...`: Provide custom keywords to prioritize in the paper list.
    *   `--ocean`: Use the predefined list of ocean science keywords for prioritization. (`--keywords` takes precedence if both are used).
    
    Examples:
    ```bash
    # Default behavior
    python arxiv_summarizer.py 

    # Fetch only AI and ML, prioritize papers mentioning 'transformer' or 'gan'
    python arxiv_summarizer.py --categories cs.AI cs.LG --keywords transformer gan

    # Use default categories, prioritize ocean science papers
    python arxiv_summarizer.py --ocean
    ```

3.  Follow the interactive prompts to select the papers to process (e.g., all found, or a limited number per category).
4.  Check the `output/` directory for the `arxiv_summary_YYYY-MM-DD.md` file.

## Customization

-   **Default Categories:** Modify the `default=[...]` list in the `add_argument('--categories', ...)` call within `main()`.
-   **Predefined Ocean Keywords:** Modify the hardcoded list assigned when `args.ocean` is true within `main()`.
-   **Fetch Size:** Adjust `FETCH_BATCH_SIZE` if needed (e.g., if consistently missing papers on high-volume days).
-   **Models/Prompts:** Update the OpenAI model names or system/user prompts in the `summarize_text` and `summarize_category` functions.

## Next Steps / Potential Improvements

-   Add non-interactive mode flags (e.g., `--process-all`, `--limit-per-category N`) to bypass the interactive prompts for automation.
-   Implement API pagination for `fetch_arxiv_papers` for robustness on very high-volume days.
-   Improve error handling (e.g., for OpenAI rate limits, network issues).
-   Make keyword lists configurable via a separate file.
-   Set up automated daily execution (e.g., using `cron` on Linux/macOS or Task Scheduler on Windows). 