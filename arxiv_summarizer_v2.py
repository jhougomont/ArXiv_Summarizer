#!/usr/bin/env python3
"""
arXiv Paper Summarizer v2.0

Refactored version with modular architecture, improved error handling,
concurrent processing, and comprehensive logging.
"""

import sys
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.arxiv_client import ArxivClient
from src.summarizer import PaperSummarizer
from src.output_writer import OutputWriter
from src.processor import PaperProcessor
from src.utils import setup_logging, validate_categories, get_user_choice, limit_papers_per_category

logger = logging.getLogger(__name__)


def validate_date(date_string: str) -> datetime.datetime:
    """Parse a YYYY-MM-DD string into a datetime object.

    Args:
        date_string: Date in YYYY-MM-DD format

    Returns:
        datetime object

    Raises:
        argparse.ArgumentTypeError: If date format is invalid
    """
    try:
        return datetime.datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_string}'. Please use YYYY-MM-DD."
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fetch and summarize arXiv papers from a specified date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default behavior (fetch yesterday's papers, default categories)
  %(prog)s

  # Fetch papers for a specific date
  %(prog)s --date 2024-05-20

  # Use custom categories and keywords
  %(prog)s --categories cs.AI cs.LG --keywords transformer attention

  # Automatically process all papers without prompts
  %(prog)s --auto-all

  # Limit to 10 papers per category
  %(prog)s --limit 10

  # Use ocean science keywords
  %(prog)s --ocean

  # Adjust logging verbosity
  %(prog)s --log-level DEBUG
        """
    )

    parser.add_argument(
        '--date',
        type=validate_date,
        help='Target date to fetch papers for (YYYY-MM-DD). Defaults to yesterday.'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        help='Specify which arXiv categories to fetch (e.g., cs.AI cs.LG). '
             'Defaults to cs.AI, cs.CV, cs.CY, cs.LG.'
    )

    parser.add_argument(
        '--keywords',
        nargs='+',
        help='One or more keywords to prioritize (case-insensitive). '
             'Papers matching these keywords in title/abstract will be listed first.'
    )

    parser.add_argument(
        '--ocean',
        action='store_true',
        help='Use a predefined list of ocean science keywords for prioritization.'
    )

    parser.add_argument(
        '--auto-all',
        action='store_true',
        help='Automatically process all papers without interactive prompt.'
    )

    parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit to N papers per category (bypasses interactive prompt).'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml).'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level (overrides config file).'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )

    return parser.parse_args()


def determine_keywords(
    args: argparse.Namespace,
    config: Config
) -> Optional[List[str]]:
    """Determine which keywords to use for prioritization.

    Args:
        args: Parsed command-line arguments
        config: Configuration object

    Returns:
        List of keywords or None if no prioritization
    """
    if args.ocean:
        keywords = config.keywords.get('ocean_science', [])
        logger.info(f"Using predefined ocean science keywords: {keywords}")
        return keywords
    elif args.keywords:
        logger.info(f"Using custom keywords: {args.keywords}")
        return args.keywords
    else:
        logger.info("No keyword prioritization specified")
        return None


def main():
    """Main entry point for the arXiv summarizer."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        config = Config()

    # Override log level if specified
    if args.log_level:
        config.logging.level = args.log_level

    # Setup logging
    setup_logging(config.logging)
    logger.info("=" * 60)
    logger.info("arXiv Paper Summarizer v2.0")
    logger.info("=" * 60)

    # Validate API key
    if not config.openai.api_key:
        logger.error("OPENAI_API_KEY environment variable is not set!")
        logger.error("Please set it in your .env file or environment.")
        sys.exit(1)

    # Determine target date
    if args.date:
        target_utc = args.date.replace(tzinfo=datetime.timezone.utc)
        logger.info(f"Using specified target date: {target_utc.strftime('%Y-%m-%d')}")
    else:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        target_utc = now_utc - datetime.timedelta(days=1)
        logger.info(f"No date specified, defaulting to yesterday (UTC): {target_utc.strftime('%Y-%m-%d')}")

    target_date = target_utc.date()

    # Determine categories
    if args.categories:
        categories = args.categories
        if not validate_categories(categories):
            logger.warning("Some category codes may be invalid. Proceeding anyway...")
    else:
        categories = config.arxiv.default_categories

    logger.info(f"Categories: {', '.join(categories)}")

    # Determine keywords
    keywords = determine_keywords(args, config)

    # Initialize components
    logger.info("Initializing components...")

    try:
        arxiv_client = ArxivClient(
            fetch_batch_size=config.arxiv.fetch_batch_size
        )

        summarizer = PaperSummarizer(
            api_key=config.openai.api_key,
            model=config.openai.default_model,
            temperature=config.openai.temperature,
            max_tokens_abstract=config.openai.max_tokens_abstract,
            max_tokens_category=config.openai.max_tokens_category,
            rate_limit_delay=config.openai.rate_limit_delay,
            max_retries=config.openai.max_retries,
            retry_delay=config.openai.retry_delay,
            concurrent_requests=config.processing.concurrent_requests
        )

        output_writer = OutputWriter(
            output_directory=config.output.directory,
            include_statistics=config.output.include_statistics,
            include_metadata=config.output.include_metadata
        )

        processor = PaperProcessor(
            arxiv_client=arxiv_client,
            summarizer=summarizer,
            output_writer=output_writer,
            max_summaries_per_category=config.processing.max_summaries_per_category
        )

    except ValueError as e:
        logger.error(f"Initialization error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        sys.exit(1)

    # Fetch papers
    logger.info("Fetching papers from arXiv API...")
    try:
        all_papers = arxiv_client.fetch_papers(categories)
        papers_found = arxiv_client.filter_by_date(
            all_papers,
            target_date,
            categories
        )
    except Exception as e:
        logger.error(f"Failed to fetch papers: {e}")
        sys.exit(1)

    num_found = len(papers_found)
    logger.info(f"Found {num_found} papers for {target_date}")

    if num_found == 0:
        logger.warning("No papers found for the specified date and categories")
        output_file, _ = processor.process(
            target_date=target_date,
            categories=categories,
            papers_to_process=[],
            all_papers=all_papers,
            keywords=keywords
        )
        logger.info(f"Empty summary written to {output_file}")
        return

    # Determine which papers to process
    papers_to_process = papers_found

    # Handle automated vs interactive mode
    if args.limit is not None:
        # Use limit from command line
        logger.info(f"Limiting to {args.limit} papers per category")
        papers_to_process = limit_papers_per_category(
            papers_found,
            categories,
            args.limit
        )
    elif not args.auto_all:
        # Interactive mode
        choice, limit = get_user_choice(num_found, allow_auto=False)

        if choice == 'quit':
            logger.info("User chose to quit. Exiting.")
            sys.exit(0)
        elif choice == 'number' and limit is not None:
            papers_to_process = limit_papers_per_category(
                papers_found,
                categories,
                limit
            )
        # If choice == 'all', papers_to_process is already set to papers_found

    logger.info(f"Processing {len(papers_to_process)} papers")

    # Process papers
    try:
        output_file, statistics = processor.process(
            target_date=target_date,
            categories=categories,
            papers_to_process=papers_to_process,
            all_papers=all_papers,
            keywords=keywords
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Processing Complete!")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")
        logger.info(f"Papers processed: {len(papers_to_process)}")
        logger.info(f"Processing time: {statistics['processing_time']:.1f}s")
        logger.info(f"API calls made: {statistics['api_calls']}")

        if keywords and statistics['keyword_matches'] > 0:
            logger.info(
                f"Keyword matches: {statistics['keyword_matches']} "
                f"({statistics['keyword_match_rate']:.1f}%)"
            )

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
