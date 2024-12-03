# pipelines.py

import json

class SaveToJSONPipeline:
    """Pipeline that saves scraped data to a JSON file."""

    def open_spider(self, spider):
        """Open the file when the spider starts."""
        self.file = open('scraped_results.json', 'w', encoding='utf-8')

    def close_spider(self, spider):
        """Close the file when the spider finishes."""
        self.file.close()

    def process_item(self, item, spider):
        """Write each scraped item to the JSON file."""
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(line)
        return item


class ContentCleanupPipeline:
    """Pipeline to clean up content by removing unnecessary whitespace."""

    def process_item(self, item, spider):
        """Clean up content field."""
        item['content'] = ' '.join(item['content'].split())  # Clean up content by removing extra spaces
        return item
    

class DisplayResultsPipeline:
    """Pipeline that formats and prints the search results in a Google-like format."""
    
    def open_spider(self, spider):
        """Initialize an empty results list when the spider starts."""
        self.results = []

    def process_item(self, item, spider):
        """Store the item in the results list."""
        self.results.append({
            'title': item['title'],
            'summary': item['content'],
            'link': item['link']
        })
        return item

    def close_spider(self, spider):
        """Print out the formatted results when the spider finishes."""
        print("\nTop 10 Related Links for the Search Query:")
        for i, result in enumerate(self.results[:10], start=1):
            print(f"{i}. {result['title']}\n   {result['summary'][:200]}...\n   {result['link']}\n")

