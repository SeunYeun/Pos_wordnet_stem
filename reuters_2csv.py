from bs4 import BeautifulSoup
import pandas as pd
import os

def extract_articles_from_sgm(filepath):
    with open(filepath, "r", encoding="ISO-8859-1") as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')
    articles = []

    for article in soup.find_all('reuters'):
        doc_id = article['newid']
        title = article.title.get_text(strip=True) if article.title else ""
        body = article.body.get_text(strip=True) if article.body else ""
        topics = [t.get_text() for t in article.topics.find_all("d")] if article.topics else []

        articles.append({
            "id": doc_id,
            "title": title,
            "body": body,
            "topics": ";".join(topics)
        })

    return articles

all_articles = []
sgm_dir = "Data/DataSet/reuters21578/"

for i in range(22):
    filename = f"reut2-{i:03d}.sgm"  
    filepath = os.path.join(sgm_dir, filename)
    print(f"📄 Đang đọc: {filename}")
    articles = extract_articles_from_sgm(filepath)
    all_articles.extend(articles)

df = pd.DataFrame(all_articles)

df.to_csv("Data/csv/reuters_all.csv", index=False)
print(f"\n✅ Đã đọc {len(df)} bài viết từ 21 tệp.")
