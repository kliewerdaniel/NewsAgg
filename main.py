from typing import List, Dict
import feedparser
import asyncio
import re
from ollama import AsyncClient
from playwright.async_api import async_playwright
import datetime
import json


class NewsSynthesizer:
    def __init__(self):
        self.llm = AsyncClient()
        self.rss_feeds = [
            ('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml'),
            ('Reuters', 'http://feed.reuters.com/reuters/topNews'),
            ('AP News', 'https://apnews.com/feed')
        ]
        self.llm_model = "vanilj/Phi-4:latest"

    async def _scrape_article(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, timeout=60000, wait_until='domcontentloaded')
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
        all_articles = []
        for feed_name, feed_url in self.rss_feeds:
            try:
                feed_data = feedparser.parse(feed_url)
                for entry in feed_data.entries[:5]:
                    content = await self._scrape_article(entry.link)
                    if content and len(content) > 100:
                        all_articles.append({
                            'title': entry.title,
                            'source': feed_name,
                            'content': content[:2000]
                        })
            except Exception as e:
                print(f"Error processing feed {feed_name}: {str(e)}")

        if not all_articles:
            return []

        cluster_prompt = f"""Group these news titles by similarity:
        {[a['title'] for a in all_articles]}
        Return JSON with groups of indices like {{"groups": [[0,1], [2,3]]}}"""
        
        # Add JSON parsing with error handling
        try:
            clusters = await self._llm_generate(cluster_prompt, format='json')
            clusters = json.loads(clusters)  # Convert JSON string to dict
        except json.JSONDecodeError:
            clusters = {'groups': [[i] for i in range(len(all_articles))]}
        
        valid_clusters = clusters.get('groups', [[i] for i in range(len(all_articles))])

        final_reports = []
        for group in valid_clusters:
            try:
                articles = [all_articles[i] for i in group if i < len(all_articles)]
                analysis = await self.analyze_articles([a['content'] for a in articles])
                report = await self.generate_unified_report(analysis)
                
                if report:
                    final_reports.append({
                        'headline': articles[0]['title'],
                        'sources': list({a['source'] for a in articles}),
                        'report': report
                    })
            except Exception as e:
                print(f"Error processing cluster: {str(e)}")

        return final_reports

        cluster_prompt = f"""Group these news titles by similarity:
        {[a['title'] for a in all_articles]}
        Return JSON with groups of indices like {{"groups": [[0,1], [2,3]]}}"""
        
        clusters = await self._llm_generate(cluster_prompt, format='json')
        valid_clusters = clusters.get('groups', [[i] for i in range(len(all_articles))])

        final_reports = []
        for group in valid_clusters:
            try:
                articles = [all_articles[i] for i in group if i < len(all_articles)]
                analysis = await self.analyze_articles([a['content'] for a in articles])
                report = await self.generate_unified_report(analysis)
                
                if report:
                    final_reports.append({
                        'headline': articles[0]['title'],
                        'sources': list({a['source'] for a in articles}),
                        'report': report
                    })
            except Exception as e:
                print(f"Error processing cluster: {str(e)}")

        return final_reports

    async def analyze_articles(self, articles: List[str]) -> Dict:
        analysis_prompt = f"""Analyze these articles with strict JSON formatting:
        {articles}
        Output JSON with:
        - facts: List of verified facts
        - conflicts: Detailed conflict analysis
        - missing_info: Specific missing details
        - entities: Key people/organizations
        - summary: 100-word overview"""
        return await self._llm_generate(analysis_prompt, format='json')

    async def generate_unified_report(self, analysis: Dict) -> str:
        for attempt in range(3):
            try:
                response = await self._llm_generate(f"""Create comprehensive report from:
                {analysis}
                Use Markdown formatting with these headers:
                ## Verified Facts
                ## Conflict Analysis  
                ## Research Needed
                ## Conclusion (300+ words)
                Include bullet points and clear section spacing.""")
                if self._validate_report(response):
                    return response
            except Exception as e:
                print(f"Generation attempt {attempt+1} failed: {str(e)}")
            await asyncio.sleep(2 ** attempt)
        return "⚠️ Report generation failed"

    def _validate_report(self, text: str) -> bool:
        return all(section in text for section in ['## Verified Facts', '## Conflict Analysis', '## Conclusion'])

    async def _llm_generate(self, prompt: str, format: str = None, retries=3):
        for attempt in range(retries):
            try:
                params = {
                    'model': self.llm_model,
                    'prompt': prompt[:15000],
                    'options': {'temperature': 0.3 + attempt*0.2}
                }
                if format == 'json':
                    params['format'] = 'json'

                response = await self.llm.generate(**params)
                return response.get('response', '')
            except Exception as e:
                print(f"LLM Error (attempt {attempt+1}): {str(e)}")
                await asyncio.sleep(1.5 ** attempt)
        return ""

def create_slug(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = slug.replace(' ', '-').replace('--', '-')
    return slug[:50]

async def main():
    synthesizer = NewsSynthesizer()
    reports = await synthesizer.process_feeds()
    
    if not reports:
        print("No reports generated")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"news_report_{timestamp}.md"
    
    # Prepare conclusions and slugs
    key_conclusions = []
    slugs = []
    for i, report in enumerate(reports, 1):
        conclusion_match = re.search(r'## Conclusion\n+(.*?)(?=\n##|\Z)', 
                                   report['report'], re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else "Conclusion not available."
        key_conclusions.append({
            'headline': report['headline'],
            'sources': report['sources'],
            'conclusion': conclusion
        })
        slugs.append(create_slug(f"report-{i}-{report['headline']}"))

    # Build Markdown content
    md_content = [
        "# 🌐 Daily News Synthesis Report",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**AI Model:** {synthesizer.llm_model}  ",
        f"**News Sources:** {', '.join([f[0] for f in synthesizer.rss_feeds])}  ",
        "\n---\n",
        "## 📑 Table of Contents",
        "- [News Reports](#news-reports)",
        *[f"  - [Report {i}: {r['headline']}](#{slugs[i-1]})" for i, r in enumerate(reports, 1)],
        "- [🔑 Key Conclusions](#key-conclusions)",
        "\n---\n",
        "## 📰 News Reports"
    ]

    # Add individual reports
    for i, (report, slug) in enumerate(zip(reports, slugs), 1):
        formatted_report = re.sub(r'##\s+(.+?)\n', r'### \1\n', report['report'])
        md_content += [
            f"\n<a id='{slug}'></a>",
            f"### Report {i}: {report['headline']}",
            f"**Sources:** {', '.join(report['sources'])}  ",
            f"**Word Count:** {len(report['report'].split())} words  ",
            "\n" + formatted_report,
            "\n[Back to Contents](#-table-of-contents)",
            "\n---"
        ]

    # Add key conclusions
    md_content += [
        "\n<a id='key-conclusions'></a>",
        "## 🔑 Key Conclusions",
        *[f"\n### {concl['headline']}\n"
          f"*Sources: {', '.join(concl['sources'])}*\n"
          f"{concl['conclusion']}\n"
          f"[View full report](#{slugs[i]})\n---" 
          for i, concl in enumerate(key_conclusions)]
    ]

    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

    print(f"Report generated: {filename}")

if __name__ == "__main__":
    asyncio.run(main())