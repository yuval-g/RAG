#!/usr/bin/env python3
"""
Web Scraper RAG Example

This example demonstrates how to build a RAG system that can scrape web content
and answer questions based on the scraped information.
"""

import sys
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import time
from bs4 import BeautifulSoup
import re

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_engine.core.engine import RAGEngine
from rag_engine.core.config import PipelineConfig
from rag_engine.core.models import Document


class WebScraperRAG:
    """A RAG system that can scrape and index web content"""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the web scraper RAG system"""
        self.config = config or PipelineConfig(
            llm_model="gemini-1.5-flash",
            temperature=0.2,
            chunk_size=800,
            chunk_overlap=100,
            retrieval_k=5
        )
        self.engine = RAGEngine(self.config)
        self.scraped_urls = set()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def scrape_url_sync(self, url: str) -> Optional[Document]:
        """Scrape a single URL synchronously"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return self._parse_html_content(url, response.text)
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    async def scrape_url_async(self, url: str) -> Optional[Document]:
        """Scrape a single URL asynchronously"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    return self._parse_html_content(url, html_content)
                else:
                    print(f"❌ HTTP {response.status} for {url}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    def _parse_html_content(self, url: str, html_content: str) -> Optional[Document]:
        """Parse HTML content and extract text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            # Try to find main content areas
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '.post', '.entry']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text
            text = main_content.get_text()
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < 100:  # Skip pages with very little content
                return None
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": title_text,
                "domain": urlparse(url).netloc,
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(text),
                "source_type": "web_scrape"
            }
            
            # Try to extract description
            description_meta = soup.find('meta', attrs={'name': 'description'})
            if description_meta:
                metadata["description"] = description_meta.get('content', '')
            
            # Try to extract keywords
            keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_meta:
                metadata["keywords"] = keywords_meta.get('content', '')
            
            doc_id = f"web_{hash(url) % 100000}"
            
            return Document(
                content=text,
                metadata=metadata,
                doc_id=doc_id
            )
            
        except Exception as e:
            print(f"❌ Error parsing HTML for {url}: {e}")
            return None
    
    def scrape_urls(self, urls: List[str], max_concurrent: int = 5) -> List[Document]:
        """Scrape multiple URLs"""
        if len(urls) == 1:
            # Single URL - use sync method
            doc = self.scrape_url_sync(urls[0])
            return [doc] if doc else []
        
        # Multiple URLs - use async method
        return asyncio.run(self._scrape_urls_async(urls, max_concurrent))
    
    async def _scrape_urls_async(self, urls: List[str], max_concurrent: int) -> List[Document]:
        """Scrape multiple URLs asynchronously"""
        async with self:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def scrape_with_semaphore(url):
                async with semaphore:
                    return await self.scrape_url_async(url)
            
            tasks = [scrape_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            documents = []
            for result in results:
                if isinstance(result, Document):
                    documents.append(result)
                elif isinstance(result, Exception):
                    print(f"❌ Scraping error: {result}")
            
            return documents
    
    def scrape_and_index(self, urls: List[str]) -> bool:
        """Scrape URLs and add to the knowledge base"""
        print(f"🕷️  Scraping {len(urls)} URLs...")
        
        documents = self.scrape_urls(urls)
        
        if documents:
            print(f"✅ Successfully scraped {len(documents)} pages")
            
            # Add to RAG engine
            result = self.engine.add_documents(documents)
            if result:
                self.scraped_urls.update(doc.metadata["url"] for doc in documents)
                print(f"📚 Added {len(documents)} documents to knowledge base")
                return True
        
        print("❌ No documents were scraped successfully")
        return False
    
    def discover_links(self, base_url: str, max_depth: int = 1, 
                      same_domain_only: bool = True) -> List[str]:
        """Discover links from a base URL"""
        discovered_urls = set()
        to_visit = [(base_url, 0)]
        visited = set()
        
        base_domain = urlparse(base_url).netloc
        
        while to_visit:
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited or depth > max_depth:
                continue
            
            visited.add(current_url)
            
            try:
                print(f"🔍 Discovering links from {current_url} (depth {depth})")
                
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    # Parse URL
                    parsed = urlparse(full_url)
                    
                    # Skip non-HTTP URLs
                    if parsed.scheme not in ['http', 'https']:
                        continue
                    
                    # Skip same domain check if required
                    if same_domain_only and parsed.netloc != base_domain:
                        continue
                    
                    # Skip common non-content URLs
                    if any(skip in full_url.lower() for skip in [
                        '.pdf', '.jpg', '.png', '.gif', '.css', '.js',
                        'mailto:', 'tel:', '#', 'javascript:'
                    ]):
                        continue
                    
                    discovered_urls.add(full_url)
                    
                    # Add to visit queue if within depth limit
                    if depth < max_depth:
                        to_visit.append((full_url, depth + 1))
                
            except Exception as e:
                print(f"❌ Error discovering links from {current_url}: {e}")
        
        return list(discovered_urls)
    
    def query_web_knowledge(self, question: str, strategy: str = "basic") -> Dict[str, Any]:
        """Query the web-based knowledge base"""
        try:
            response = self.engine.query(question, strategy=strategy)
            
            result = {
                "question": question,
                "answer": response.answer,
                "confidence": response.confidence_score,
                "processing_time": response.processing_time,
                "web_sources": []
            }
            
            # Format web sources
            for doc in response.source_documents:
                if doc.metadata.get("source_type") == "web_scrape":
                    source_info = {
                        "title": doc.metadata.get("title", "Unknown"),
                        "url": doc.metadata.get("url", "Unknown"),
                        "domain": doc.metadata.get("domain", "Unknown"),
                        "scraped_at": doc.metadata.get("scraped_at", "Unknown"),
                        "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    }
                    result["web_sources"].append(source_info)
            
            return result
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e),
                "web_sources": []
            }
    
    def get_scraped_stats(self) -> Dict[str, Any]:
        """Get statistics about scraped content"""
        system_info = self.engine.get_system_info()
        
        # Count domains
        domains = {}
        total_content_length = 0
        
        # This would require access to document metadata
        # For now, return basic stats
        return {
            "total_documents": system_info["stats"]["indexed_documents"],
            "scraped_urls_count": len(self.scraped_urls),
            "system_ready": system_info["stats"]["retriever_ready"]
        }


def demo_wikipedia_scraping():
    """Demo scraping Wikipedia articles"""
    print("📖 Wikipedia Scraping Demo")
    print("=" * 50)
    
    # Wikipedia URLs about AI topics
    wikipedia_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Computer_vision"
    ]
    
    # Initialize scraper
    scraper = WebScraperRAG()
    
    # Scrape and index
    if scraper.scrape_and_index(wikipedia_urls):
        print("\n✅ Wikipedia articles indexed successfully!")
        
        # Test questions
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of computer vision?",
            "What is natural language processing used for?",
            "What are neural networks in deep learning?"
        ]
        
        print(f"\n🤖 Testing with {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Q: {question}")
            result = scraper.query_web_knowledge(question, strategy="multi_query")
            
            print(f"   A: {result['answer'][:200]}...")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {len(result['web_sources'])} web pages")
            
            if result['web_sources']:
                for source in result['web_sources'][:2]:  # Show first 2 sources
                    print(f"     - {source['title']} ({source['domain']})")
        
        # Show stats
        stats = scraper.get_scraped_stats()
        print(f"\n📊 Final Stats:")
        print(f"Documents indexed: {stats['total_documents']}")
        print(f"URLs scraped: {stats['scraped_urls_count']}")
    
    else:
        print("❌ Failed to scrape Wikipedia articles")


def demo_news_scraping():
    """Demo scraping news articles (example URLs)"""
    print("📰 News Scraping Demo")
    print("=" * 50)
    
    # Example news URLs (replace with actual URLs)
    news_urls = [
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com/category/artificial-intelligence/",
        "https://www.wired.com/tag/artificial-intelligence/"
    ]
    
    print("⚠️  Note: This demo uses example URLs. Replace with actual news URLs.")
    print("Some sites may block scraping or require special handling.")
    
    scraper = WebScraperRAG()
    
    # Try to scrape (may fail due to anti-bot measures)
    print(f"\n🕷️  Attempting to scrape {len(news_urls)} news sources...")
    
    for url in news_urls:
        print(f"Trying: {url}")
        doc = scraper.scrape_url_sync(url)
        if doc:
            print(f"✅ Successfully scraped: {doc.metadata['title'][:50]}...")
        else:
            print(f"❌ Failed to scrape: {url}")


def demo_link_discovery():
    """Demo link discovery from a base URL"""
    print("🔗 Link Discovery Demo")
    print("=" * 50)
    
    base_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    scraper = WebScraperRAG()
    
    print(f"🔍 Discovering links from: {base_url}")
    
    # Discover links (depth 1 to avoid too many URLs)
    discovered_urls = scraper.discover_links(base_url, max_depth=1, same_domain_only=True)
    
    print(f"✅ Discovered {len(discovered_urls)} URLs")
    
    # Show first 10 URLs
    print("\n📋 Sample discovered URLs:")
    for i, url in enumerate(discovered_urls[:10], 1):
        print(f"{i}. {url}")
    
    if len(discovered_urls) > 10:
        print(f"... and {len(discovered_urls) - 10} more")
    
    # Optionally scrape a few of them
    if discovered_urls:
        print(f"\n🕷️  Scraping first 3 discovered URLs...")
        sample_urls = discovered_urls[:3]
        
        if scraper.scrape_and_index(sample_urls):
            print("✅ Successfully indexed discovered content!")
            
            # Test a question
            result = scraper.query_web_knowledge(
                "What topics are related to artificial intelligence?",
                strategy="rag_fusion"
            )
            
            print(f"\n💡 Answer: {result['answer'][:300]}...")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"🔗 Web sources: {len(result['web_sources'])}")


def interactive_web_rag():
    """Interactive web RAG session"""
    print("🌐 Interactive Web RAG System")
    print("=" * 50)
    
    scraper = WebScraperRAG()
    
    print("Commands:")
    print("  scrape <url1> <url2> ... - Scrape and index URLs")
    print("  discover <base_url> - Discover and scrape links from base URL")
    print("  ask <question> - Ask a question")
    print("  stats - Show statistics")
    print("  quit - Exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🌐 Command: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = scraper.get_scraped_stats()
                print(f"📊 Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower().startswith('scrape '):
                urls = user_input[7:].split()
                if urls:
                    print(f"🕷️  Scraping {len(urls)} URLs...")
                    success = scraper.scrape_and_index(urls)
                    if success:
                        print("✅ Scraping completed!")
                    else:
                        print("❌ Scraping failed!")
                else:
                    print("❌ Please provide URLs to scrape")
                continue
            
            if user_input.lower().startswith('discover '):
                base_url = user_input[9:].strip()
                if base_url:
                    print(f"🔍 Discovering links from {base_url}...")
                    discovered = scraper.discover_links(base_url, max_depth=1)
                    print(f"Found {len(discovered)} URLs")
                    
                    if discovered:
                        # Scrape first 5
                        sample_urls = discovered[:5]
                        print(f"Scraping first {len(sample_urls)} URLs...")
                        success = scraper.scrape_and_index(sample_urls)
                        if success:
                            print("✅ Discovery and scraping completed!")
                else:
                    print("❌ Please provide a base URL")
                continue
            
            if user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    print("🤔 Processing question...")
                    result = scraper.query_web_knowledge(question, strategy="multi_query")
                    
                    print(f"\n💡 Answer: {result['answer']}")
                    print(f"📊 Confidence: {result['confidence']:.2f}")
                    
                    if result['web_sources']:
                        print(f"\n🔗 Web Sources:")
                        for i, source in enumerate(result['web_sources'], 1):
                            print(f"  {i}. {source['title']}")
                            print(f"     {source['url']}")
                else:
                    print("❌ Please provide a question")
                continue
            
            print("❓ Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Scraper RAG Example")
    parser.add_argument("--mode", choices=["wikipedia", "news", "discovery", "interactive"],
                       default="wikipedia", help="Demo mode to run")
    parser.add_argument("--urls", nargs="*", help="URLs to scrape")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "wikipedia":
            demo_wikipedia_scraping()
        elif args.mode == "news":
            demo_news_scraping()
        elif args.mode == "discovery":
            demo_link_discovery()
        elif args.mode == "interactive":
            interactive_web_rag()
        
        if args.urls:
            print(f"\n🕷️  Custom URL scraping: {args.urls}")
            scraper = WebScraperRAG()
            if scraper.scrape_and_index(args.urls):
                print("✅ Custom URLs scraped successfully!")
                
                # Test with a question
                result = scraper.query_web_knowledge("What is this content about?")
                print(f"💡 Summary: {result['answer'][:200]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)