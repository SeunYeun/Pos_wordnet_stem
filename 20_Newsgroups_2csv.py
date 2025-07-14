import os
import re
import pandas as pd

def parse_newsgroup_file_minimal(file_path):
    """
    Phân tích một file bài báo và chỉ trích xuất Newsgroups, Subject, và Content.
    """
    with open(file_path, 'r', errors='ignore', encoding='utf-8') as f:
        content_raw = f.read()

    data = {
        'Newsgroups': '',
        'Subject': '',
        'Content': ''
    }

    # Trích xuất Newsgroups
    newsgroups_match = re.search(r'^Newsgroups:\s*(.*)$', content_raw, re.MULTILINE)
    if newsgroups_match:
        data['topics'] = newsgroups_match.group(1).strip()
    
    # Trích xuất Subject
    subject_match = re.search(r'^Subject:\s*(.*)$', content_raw, re.MULTILINE)
    if subject_match:
        data['title'] = subject_match.group(1).strip()

    # Tìm vị trí bắt đầu của nội dung bài báo
    lines = content_raw.split('\n')
    body_start_index = 0
    for i, line in enumerate(lines):
        if not line.strip():
            body_start_index = i + 1
            break
            
    data['body'] = '\n'.join(lines[body_start_index:]).strip()

    return data

def convert_newsgroups_to_csv_minimal(root_dir, output_csv_path):
    all_articles = []
    article_id = 0 

    for category_name in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_name)
        
        if os.path.isdir(category_path):
            print(f"Đang xử lý thư mục: {category_name}")
            
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                
                if os.path.isfile(file_path):
                    article_data = parse_newsgroup_file_minimal(file_path)
                    
                    article_data['id'] = article_id
                    all_articles.append(article_data)
                    article_id += 1 
    
    df = pd.DataFrame(all_articles)
    df = df[['id', 'topics', 'title', 'body']] 

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nĐã hoàn thành. Dữ liệu đã được lưu vào: {output_csv_path}")

if __name__ == '__main__':
    root_directory = 'Data/DataSet/20_newsgroups' 
    output_csv_file = 'Data/csv/20_newsgroups_data.csv'

    if not os.path.isdir(root_directory):
        print(f"Lỗi: Thư mục '{root_directory}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        print("Hãy đảm bảo 'root_directory' trỏ đến thư mục chứa 20 folder con của newsgroup.")
    else:
        convert_newsgroups_to_csv_minimal(root_directory, output_csv_file)