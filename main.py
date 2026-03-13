"""
Unified scraper entrypoint.
"""
import scraper_arxiv
import scraper_github
import scraper_huggingface


def main():
    print("\n🚀 RUNNING ALL SCRAPERS\n")
    # scraper_arxiv.validate_and_backfill()
    # scraper_arxiv.run()
    # scraper_github.run()
    scraper_huggingface.run()
    print("\n🎉 ALL SCRAPERS COMPLETE\n")


if __name__ == "__main__":
    main()
