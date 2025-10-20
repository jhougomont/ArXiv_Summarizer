# arXiv Paper Summarizer v2.0

A powerful, production-ready tool for fetching and summarizing academic papers from arXiv with AI-powered summaries.

## What's New in v2.0? 🚀

- **5x Faster**: Concurrent API calls process papers in parallel
- **More Reliable**: Automatic retry with exponential backoff for API failures
- **Better Organized**: Modular architecture with clean separation of concerns
- **Fully Configurable**: YAML-based configuration file
- **Production Ready**: Comprehensive logging, error handling, and testing
- **Automation Friendly**: New CLI flags for scripting and cron jobs
- **Type Safe**: Full type hints throughout the codebase

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd arxiv_summary

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Basic Usage

```bash
# Run with default settings (yesterday's papers)
python arxiv_summarizer_v2.py

# Process specific date
python arxiv_summarizer_v2.py --date 2024-05-20

# Automatic mode (no interactive prompts)
python arxiv_summarizer_v2.py --auto-all

# Limit papers per category
python arxiv_summarizer_v2.py --limit 10

# Use ocean science keywords
python arxiv_summarizer_v2.py --ocean

# Custom categories and keywords
python arxiv_summarizer_v2.py --categories cs.AI cs.LG --keywords transformer attention
```

## Features

### Core Functionality

- **Intelligent Fetching**: Retrieves papers from arXiv API with configurable categories
- **Date Filtering**: Processes papers published or updated on specific dates
- **AI Summarization**: Uses OpenAI GPT-3.5-turbo to generate concise summaries
- **Category Summaries**: Generates overview summaries for each category with paper citations
- **Keyword Prioritization**: Highlights and sorts papers matching specific keywords
- **Rich Output**: Generates detailed Markdown files with statistics

### Technical Features

- **Concurrent Processing**: Process multiple papers simultaneously (configurable workers)
- **Error Handling**: Retry logic with exponential backoff for rate limits and failures
- **Logging**: Comprehensive logging to both file and console
- **Configuration**: YAML-based config file for all settings
- **Testing**: Unit tests for core functionality
- **Type Hints**: Full type annotations for better IDE support

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
arxiv:
  fetch_batch_size: 500
  default_categories:
    - cs.AI
    - cs.CV
    - cs.CY
    - cs.LG

openai:
  default_model: gpt-3.5-turbo
  temperature: 0.5
  max_tokens_abstract: 100
  max_tokens_category: 200
  rate_limit_delay: 0.5
  max_retries: 3
  concurrent_requests: 5

keywords:
  ocean_science:
    - ocean
    - marine
    - sea
  # Add your own keyword lists...
```

## Command-Line Options

```
--date YYYY-MM-DD          Target date (defaults to yesterday)
--categories CAT1 CAT2...  arXiv categories to fetch
--keywords KW1 KW2...      Keywords for prioritization
--ocean                    Use ocean science keywords
--auto-all                 Process all papers without prompts
--limit N                  Limit to N papers per category
--config FILE              Path to config file
--log-level LEVEL          Set logging level (DEBUG, INFO, WARNING, ERROR)
--version                  Show version information
--help                     Show help message
```

## Architecture

The v2 codebase is organized into logical modules:

```
arxiv_summary/
├── arxiv_summarizer_v2.py    # Main entry point
├── config.yaml               # Configuration file
├── src/
│   ├── config.py            # Configuration management
│   ├── arxiv_client.py      # arXiv API client
│   ├── summarizer.py        # OpenAI summarization
│   ├── output_writer.py     # Markdown output generation
│   ├── processor.py         # Main processing orchestration
│   └── utils.py             # Utility functions
├── tests/
│   ├── test_arxiv_client.py
│   ├── test_config.py
│   └── test_processor.py
└── output/                   # Generated summaries
```

## Output Format

Generated files include:

1. **Header**: Date, categories, paper counts, keywords used
2. **Statistics**: Papers by category, keyword matches, processing time
3. **Category Summaries**: AI-generated overview for each category
4. **Paper Details**: Full information for each paper including:
   - Title (with keyword match highlighting)
   - Authors
   - Category
   - Publication date
   - Links (abstract and PDF)
   - AI-generated summary

## Automation Examples

### Daily Cron Job

```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/arxiv_summary && /path/to/venv/bin/python arxiv_summarizer_v2.py --auto-all --log-level WARNING >> daily.log 2>&1
```

### Process Multiple Dates

```bash
#!/bin/bash
for i in {0..6}; do
  date=$(date -d "$i days ago" +%Y-%m-%d)  # Linux
  python arxiv_summarizer_v2.py --date $date --auto-all
done
```

## Testing

Run the test suite:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_arxiv_client -v
```

## Performance

| Papers | v1 (Sequential) | v2 (Concurrent) | Speedup |
|--------|----------------|-----------------|---------|
| 50     | ~2.5 min       | ~30 sec         | 5x      |
| 100    | ~5 min         | ~1 min          | 5x      |
| 200    | ~10 min        | ~2 min          | 5x      |

*With 5 concurrent workers and 0.5s rate limiting*

## Migration from v1

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

Quick migration:
1. Install new dependencies: `pip install -r requirements.txt`
2. Use `arxiv_summarizer_v2.py` instead of `arxiv_summarizer.py`
3. All old command-line arguments still work!

## Troubleshooting

### "No module named 'yaml'"
```bash
pip install pyyaml
```

### Rate limit errors
Increase `rate_limit_delay` in `config.yaml` or reduce `concurrent_requests`.

### Out of memory
Reduce `concurrent_requests` in `config.yaml`.

### Logs too verbose
Use `--log-level WARNING` or edit `config.yaml`.

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## API Costs

Approximate costs with GPT-3.5-turbo:
- 100 papers: ~$0.05-0.10
- 200 papers: ~$0.10-0.20

Costs vary based on abstract length and category summaries.

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Follow existing code style
3. Update documentation
4. Run tests before submitting

## License

[Your License Here]

## Support

- Report issues on GitHub
- Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for upgrade help
- Review example outputs in `output/` directory

## Acknowledgments

Built with:
- [arXiv API](https://arxiv.org/help/api)
- [OpenAI API](https://openai.com/api/)
- [feedparser](https://pypi.org/project/feedparser/)
