"""
Documentation Scraper Module

Scrapes documentation websites and extracts clean content for RAPTOR processing.
"""

from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
import time

import requests
from bs4 import BeautifulSoup
import html2text


class DocumentationScraper:
    """
    Scrapes documentation websites and extracts clean content.
    
    Features:
    - Recursive crawling with depth control
    - Smart content extraction (removes navigation, ads, etc.)
    - HTML to Markdown conversion
    - Rate limiting and respectful crawling
    - Link validation and filtering
    """
    
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 50, delay: float = 1.0):
        """
        Initialize documentation scraper.
        
        Args:
            base_url: Starting URL for documentation
            max_depth: Maximum depth to crawl (0 = single page, 1 = direct links, etc.)
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds (be respectful!)
        """
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        
        # Parse base domain
        parsed = urlparse(base_url)
        self.base_domain = f"{parsed.scheme}://{parsed.netloc}"
        self.base_path = parsed.path.rstrip('/')
        
        # Track visited URLs
        self.visited_urls: Set[str] = set()
        self.scraped_pages: List[Dict] = []
        
        # HTML to Markdown converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        
        # User agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Documentation Scraper for Educational Purposes)'
        }
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL should be crawled.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL should be crawled, False otherwise
        """
        if not url or url in self.visited_urls:
            return False
        
        parsed = urlparse(url)
        
        # Must be same domain and under base path
        if not url.startswith(self.base_domain):
            return False
        
        # Skip common non-documentation URLs
        skip_patterns = ['#', 'javascript:', 'mailto:', '.pdf', '.zip', '.jpg', '.png', '.gif', '.svg']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        return True
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Extract clean content from page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Dictionary with page metadata and content, or None if extraction failed
        """
        # Try to find main content area (common patterns)
        main_content = None
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '.documentation',
            '.doc-content',
            '.markdown-body',
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return None
        
        # Remove unwanted elements
        for element in main_content.select('nav, header, footer, aside, .sidebar, .navigation, .toc, script, style, .ad, .advertisement'):
            element.decompose()
        
        # Get title
        title = soup.find('h1')
        if title:
            title = title.get_text().strip()
        else:
            title = soup.title.string if soup.title else url
        
        # Convert to markdown
        html_content = str(main_content)
        markdown_content = self.html_converter.handle(html_content)
        
        # Clean up markdown
        lines = [line for line in markdown_content.split('\n') if line.strip()]
        markdown_content = '\n'.join(lines)
        
        # Skip if content is too short (likely not useful)
        if len(markdown_content) < 100:
            return None
        
        return {
            'url': url,
            'title': title,
            'content': markdown_content,
            'length': len(markdown_content)
        }
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Extract valid links from page.
        
        Args:
            soup: BeautifulSoup object of the page
            current_url: Current page URL for resolving relative links
            
        Returns:
            List of valid URLs to crawl
        """
        links = []
        
        for link in soup.find_all('a', href=True):
            url = urljoin(current_url, link['href'])
            # Remove fragments
            url = url.split('#')[0]
            
            if self.is_valid_url(url):
                links.append(url)
        
        return links
    
    def scrape_page(self, url: str) -> Dict:
        """
        Scrape single page.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page data, or None if scraping failed
        """
        try:
            print(f"📄 Scraping: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            page_data = self.extract_content(soup, url)
            
            if page_data and page_data['content']:
                # Extract links for crawling
                page_data['links'] = self.extract_links(soup, url)
                return page_data
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error scraping {url}: {str(e)}")
            return None
    
    def crawl(self, start_url: str = None, depth: int = 0) -> List[Dict]:
        """
        Recursively crawl documentation site.
        
        Args:
            start_url: Starting URL (defaults to base_url)
            depth: Current depth level
            
        Returns:
            List of scraped pages
        """
        if start_url is None:
            start_url = self.base_url
        
        # Check limits
        if depth > self.max_depth or len(self.scraped_pages) >= self.max_pages:
            return self.scraped_pages
        
        # Skip if already visited
        if start_url in self.visited_urls:
            return self.scraped_pages
        
        self.visited_urls.add(start_url)
        
        # Scrape page
        page_data = self.scrape_page(start_url)
        
        if page_data:
            page_data['depth'] = depth
            self.scraped_pages.append(page_data)
            
            # Crawl linked pages
            if depth < self.max_depth:
                for link in page_data.get('links', [])[:10]:  # Limit links per page
                    if len(self.scraped_pages) >= self.max_pages:
                        break
                    
                    time.sleep(self.delay)  # Be respectful
                    self.crawl(link, depth + 1)
        
        return self.scraped_pages
    
    def scrape(self) -> List[Dict]:
        """
        Main scraping method.
        
        Returns:
            List of all scraped pages with metadata
        """
        print(f"\n🚀 Starting documentation scraper...")
        print(f"   Base URL: {self.base_url}")
        print(f"   Max Depth: {self.max_depth}")
        print(f"   Max Pages: {self.max_pages}")
        print(f"   Delay: {self.delay}s\n")
        
        pages = self.crawl()
        
        print(f"\n✓ Scraping complete!")
        print(f"   Pages scraped: {len(pages)}")
        print(f"   Total content: {sum(p['length'] for p in pages):,} characters\n")
        
        return pages
