from typing import List, Dict
import feedparser
import asyncio
from ollama import AsyncClient
from playwright.async_api import async_playwright
import datetime  # Added for timestamp

class NewsSynthesizer:
    def __init__(self):
        self.llm = AsyncClient()
        self.rss_feeds = [
            ('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml'),
            ('Reuters', 'http://feeds.reuters.com/reuters/topNews'),
            ('AP News', 'https://apnews.com/feed')
        ]
        self.llm_model = "vanilj/Phi-4:latest"

    async def _scrape_article(self, url: str) -> str:
        """Improved scraping with better timeout handling"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Increased timeout to 60 seconds
                await page.goto(url, timeout=60000, wait_until='domcontentloaded')
                
                # Smart content extraction
                content = await page.evaluate('''() => {
                    const selectors = ['article', 'main', '[itemprop="articleBody"]'];
                    for (const selector of selectors) {
                        const el = document.querySelector(selector);
                        if (el?.textContent?.trim().length > 500) {
                            return el.innerText;
                        }
                    }
                    return document.body.innerText;
                }''')
                
                await browser.close()
                return content.strip()
                
        except Exception as e:
            print(f"Scraping failed for {url}: {str(e)}")
            return ""

    async def process_feeds(self):
        """Merged and fixed feed processing method"""
        all_articles = []
        
        # Process RSS feeds with error handling
        for feed_name, feed_url in self.rss_feeds:
            try:
                print(f"Processing {feed_name} feed...")
                feed_data = feedparser.parse(feed_url)
                
                for entry in feed_data.entries[:5]:  # Limit to 3 articles per feed
                    content = await self._scrape_article(entry.link)
                    if content and len(content) > 100:  # Filter empty/scraping errors
                        all_articles.append({
                            'title': entry.title,
                            'source': feed_name,  # Fixed variable name
                            'content': content[:2000]
                        })
                        
            except Exception as e:
                # Fixed error message with proper variables
                print(f"Error processing feed {feed_name} ({feed_url}): {str(e)}")

        # Early return if no articles
        if not all_articles:
            print("No articles collected. Check:")
            print("- Network connectivity")
            print("- RSS feed URLs")
            print("- Scraping error messages")
            return []

        # Improved clustering with validation
        print("\n🧩 Clustering articles...")
        cluster_prompt = f"""Group these news titles by story similarity:
        {[a['title'] for a in all_articles]}
        Return JSON with groups of indices like {{"groups": [[0,1], [2,3]]}}"""
        
        clusters = await self._llm_generate(cluster_prompt, format='json')
        
        # Validate cluster format
        valid_clusters = []
        if isinstance(clusters, dict) and 'groups' in clusters:
            for group in clusters['groups']:
                if isinstance(group, list) and all(isinstance(i, int) for i in group):
                    valid_clusters.append(group)
        
        # Fallback to individual articles if clustering failed
        if not valid_clusters:
            valid_clusters = [[i] for i in range(len(all_articles))]

        final_reports = []
        for group in valid_clusters:
                print(f"\n📦 Processing cluster {group}")
                articles = [all_articles[i] for i in group if i < len(all_articles)]
                
                print("📝 Analyzing articles...")
                analysis = await self.analyze_articles([a['content'] for a in articles])
                
                print("🖨️ Generating report...")
                report = await self.generate_unified_report(analysis)
                
                final_reports.append({
                    'headline': articles[0]['title'],
                    'sources': list({a['source'] for a in articles}),
                    'report': report
                })
                print("✅ Report generated")

        return final_reports

    async def analyze_articles(self, articles: List[str]) -> Dict:
        """Fixed LLM analysis call"""
        analysis_prompt = f"""Analyze these news articles with strict formatting:
{articles}
Output JSON with:
- facts: List of verified facts
- conflicts: Detailed conflict analysis
- missing_info: Specific missing details with research questions
- entities: Key people/organizations
- summary: 100-word overview"""
        return await self._llm_generate(analysis_prompt, format='json')

    async def generate_unified_report(self, analysis: Dict) -> str:
        """Improved generation with context management"""
        MAX_RETRIES = 3
        report = ""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._llm_generate(f"""Create comprehensive report from:
            {analysis}
            Follow this structure:
            1. **Verified Facts** (bullet points)
            2. **Conflict Analysis** (subsections if needed)
            3. **Further Research Needed** (specific questions)
            4. **Conclusion** (300+ words)
            
            Maintain consistent section lengths.""", format='json' if attempt == 0 else None)
        
                if self._validate_report(response):
                    return response
            except Exception as e:
                print(f"Report generation attempt {attempt+1} failed: {str(e)}")
            await asyncio.sleep(2 ** attempt)
    
        return "Could not generate complete report"

    def _validate_report(self, text: str) -> bool:
        sections = ["Verified Facts", "Conflict Analysis", 
               "Further Research Needed", "Conclusion"]
        return all(section in text for section in sections)

    # Update the _llm_generate method to use correct format values
    async def _llm_generate(self, prompt: str, format: str = None, retries=3):
        """Enhanced LLM communication"""
        for attempt in range(retries):
            try:
                params = {
                    'model': self.llm_model,
                    'prompt': prompt[:15000],  # Hard cap for safety
                    'options': {'temperature': 0.3 + attempt*0.2}
                }
                if format == 'json':
                    params['format'] = 'json'

                response = await self.llm.generate(**params)
                content = response.get('response', '')
            
                if content.strip():
                    return content
                
            except Exception as e:
                print(f"LLM Error (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(1.5 ** attempt)
        return "⚠️ Content generation failed after multiple attempts"    
    
    async def process_feeds(self):
        """Improved clustering with validation"""
        all_articles = []
        final_reports = []
                
        # Process RSS feeds with error handling
        for feed_name, feed_url in self.rss_feeds:
            try:
                feed_data = feedparser.parse(feed_url)  # ✅ Pass only the URL
                for entry in feed_data.entries[:5]:
                    content = await self._scrape_article(entry.link)
                    if content and len(content) > 100:  # Filter empty/scraping errors
                        all_articles.append({
                            'title': entry.title,
                            'source': feed_name,
                            'content': content[:2000]
                        })
            except Exception as e:
                print(f"⚠️ Feed processing error: {str(e)}")

        # Improved clustering with title-based grouping
        cluster_prompt = f"""Group these news titles by story similarity:
        {[a['title'] for a in all_articles]}
        Return JSON with groups of indices like {{"groups": [[0,1], [2,3]]}}"""
        
        clusters = await self._llm_generate(cluster_prompt, format='json')
        
        # Validate cluster format
        valid_clusters = []
        if isinstance(clusters, dict) and 'groups' in clusters:
            for group in clusters['groups']:
                if isinstance(group, list) and all(isinstance(i, int) for i in group):
                    valid_clusters.append(group)
        
        # Fallback to individual articles if clustering failed
        if not valid_clusters:
            valid_clusters = [[i] for i in range(len(all_articles))]

        final_reports = []
        for group in valid_clusters:
            report = ""  # Initialize report variable
            try:
                articles = [all_articles[i] for i in group if i < len(all_articles)]
                analysis = await self.analyze_articles([a['content'] for a in articles])
                report = await self.generate_unified_report(analysis)
                
                # Add content validation
                if len(report) < 400:
                    print("⚠️ Short report detected, regenerating...")
                    report = await self.generate_unified_report(analysis)
                    
                if not report.strip():
                    continue
                
                final_reports.append({
                    'headline': articles[0]['title'],
                    'sources': list({a['source'] for a in articles}),
                    'report': report
                })
            except Exception as e:
                print(f"Error processing cluster: {str(e)}")

        return final_reports

async def main():
    print("🚀 Starting news synthesizer...")
    synthesizer = NewsSynthesizer()
    reports = await synthesizer.process_feeds()
    
    if not reports:
        print("\n😞 No reports generated")
        return
    
    # Generate Markdown content
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"news_report_{timestamp}.md"
    
    md_content = [
        "# Daily News Synthesis Report",
        f"*Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"**Model:** {synthesizer.llm_model}",
        f"**Sources:** {', '.join([f[0] for f in synthesizer.rss_feeds])}",
        ""
    ]

    for i, report in enumerate(reports, 1):
        body = report['report'].replace('#', '##')  # Prevent header collisions
        md_content.extend([
            f"## Report {i}: {report['headline']}",
            f"**Sources:** {', '.join(report['sources'])}",
            f"**Word Count:** {len(report['report'].split())} words",
            "",
            body,
            "",
            "---",
            ""
        ])


    # Write to Markdown file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"\n✅ Successfully generated report: {filename}")

    # Keep existing console output
    print("\n📰 FINAL REPORTS:")
    for i, report in enumerate(reports, 1):
        print(f"\n🔷 Report {i}: {report['headline']}")
        print(f"📚 Sources: {', '.join(report['sources'])}")
        print(f"📝 Summary: {report['report']}")
        print("\n" + "-"*80)

if __name__ == "__main__":
    asyncio.run(main())