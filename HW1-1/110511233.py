import sys

import requests
from bs4 import BeautifulSoup
from tqdm import trange, tqdm
from time import sleep
import json

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import os 

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def dict_to_jsonl(dicts, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for d in dicts:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')
            
def check_valid_pic_url(url):
    return url and \
        any(url.lower().endswith(x) for x in ['.jpg', '.jpeg', '.png', '.gif']) and \
        any(url.lower().startswith(x) for x in ['http://', 'https://'])
        
def get_soup(url) -> BeautifulSoup:
    response = requests.get(url, headers = {'cookie': 'over18=1;'})
    return BeautifulSoup(response.text, "html.parser")


def crawl():   
    # page_range = (3656, 3944)
    base_url = 'https://www.ptt.cc'

    articles = []
    popular_articles = []
    
    start_index = 3640
    done = False
    
    while True:
        url = f'https://www.ptt.cc/bbs/Beauty/index{start_index}.html'
        soup = get_soup(url)
        for article in soup.find_all('div', 'r-ent'):
            date = article.find('div', 'date').text.strip()
            # 12/31 -> 1231, 1/01 -> 0101
            date = date.replace('/', '')
            date = date if len(date) == 4 else '0' + date
            
            if date == '0101':
                print(f'First page: {start_index}')
                done = True
                break
            
        if done:
            break
        
        start_index += 1

    print('Start crawling...')        
    page_num = start_index
    done = False
    while True:
        url = f'https://www.ptt.cc/bbs/Beauty/index{page_num}.html'
        soup = get_soup(url)
        for article in soup.find_all('div', 'r-ent'):
            title = article.find('div', 'title').text.strip()
            date = article.find('div', 'date').text.strip()
            url = base_url + article.find('div', 'title').a.get('href')
            
            push_text = x.text if (x := article.find('div', 'nrec').span) else ''
            push = push_text == '爆'

            # 12/31 -> 1231, 1/01 -> 0101
            date = date.replace('/', '')
            date = date if len(date) == 4 else '0' + date
            
            if page_num == start_index and date != '0101':
                continue
            elif page_num >= start_index + 10 and date == '0101':
                print(f'Last page: {page_num}')
                done = True
                break
            
            # if "[公告]" or "Fw:[公告]" in article, pass it 
            if title.startswith('[公告]') or \
                title.startswith('Fw:[公告]') or \
                title.startswith('(本文已被刪除)') or \
                title.startswith('(已被'):
                continue
            # if '公告' in title or 'Fw:[公告]' in title:
            #     continue
            
            result = {
                'date': date,
                'title': title,
                'url': url
            }
            
            articles.append(result)
            if push: 
                popular_articles.append(result)
        
        if done:
            break
        
        page_num += 1    
        sleep(0.1)
                
    dict_to_jsonl(articles, 'articles.jsonl')
    dict_to_jsonl(popular_articles, 'popular_articles.jsonl')
    
def push_thread(filtered_articles):
    push_ids, boo_ids, total_push, total_boo = {}, {}, 0, 0
    max_workers = os.cpu_count() or 1
    lock = Lock()

    def process_article(article):
        nonlocal push_ids, boo_ids, total_push, total_boo
        soup = get_soup(article['url'])
        local_push_ids, local_boo_ids = {}, {}
        local_total_push, local_total_boo = 0, 0
        
        for push in soup.find_all('div', 'push'):
            id = push.find('span', 'push-userid').text.strip()
            type = push.find('span', 'push-tag').text.strip()
            
            if type == '推':
                local_total_push += 1
                if id in local_push_ids:
                    local_push_ids[id] += 1
                else:
                    local_push_ids[id] = 1
                    
            elif type == '噓':
                local_total_boo += 1
                if id in local_boo_ids:
                    local_boo_ids[id] += 1
                else:
                    local_boo_ids[id] = 1
        
        with lock:
            total_push += local_total_push
            total_boo += local_total_boo
            for id, count in local_push_ids.items():
                push_ids[id] = push_ids.get(id, 0) + count
            for id, count in local_boo_ids.items():
                boo_ids[id] = boo_ids.get(id, 0) + count
                
        sleep(0.5)
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_article, article) for article in filtered_articles]

        for future in tqdm(futures):
            future.result()
            
    return push_ids, boo_ids, total_push, total_boo

def push(start_date, end_date):
    articles = read_jsonl('articles.jsonl')
    filtered_articles = [article for article in articles if int(start_date) <= int(article['date']) <= int(end_date)]
    
    push_ids, boo_ids, total_push, total_boo = push_thread(filtered_articles)
        
    # sort by the number of pushes, if the number of pushes is the same, sort by user_id
    sorted_push_ids = sorted(push_ids.items(), key=lambda x: (x[1], x[0]), reverse=True)
    sorted_boo_ids = sorted(boo_ids.items(), key=lambda x: (x[1], x[0]), reverse=True)    
    
    sorted_push_ids = sorted_push_ids[:10]
    sorted_boo_ids = sorted_boo_ids[:10]

    result_push_ids = [{'user_id': user_id, 'count': count} for user_id, count in sorted_push_ids]
    result_boo_ids = [{'user_id': user_id, 'count': count} for user_id, count in sorted_boo_ids]

    indent = 3 * 4
    push_top10_txt = '\n'.join([(' ' * indent + json.dumps(x) + ',') for x in result_push_ids])[indent:-1]
    boo_top10_txt = '\n'.join([(' ' * indent + json.dumps(x) + ',') for x in result_boo_ids])[indent:-1]
    
    result = {
        'push': {
            'total': total_push,
            'top10': ['push_top10_placeholder'],
        },
        'boo': {
            'total': total_boo,
            'top10': ['boo_top10_placeholder'],
        },
    }
    
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    result_json = result_json.replace('"push_top10_placeholder"', f'{push_top10_txt}')
    result_json = result_json.replace('"boo_top10_placeholder"', f'{boo_top10_txt}')

    with open(f'push_{start_date}_{end_date}.json', 'w', encoding='utf-8') as f:
        f.write(result_json)

def popular_thread(filtered_articles):
    max_workers = os.cpu_count() or 1
    img_urls = []
    lock = Lock()

    def process_article(article):
        nonlocal img_urls
        soup = get_soup(article['url'])
        urls = [x for a in soup.find_all('a') if (x := a.get('href'))]
        urls = [url for url in urls if check_valid_pic_url(url)]
        
        with lock:
            img_urls.extend(urls)
        
        sleep(0.5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_article, article) for article in filtered_articles]
        
        for future in tqdm(futures):
            future.result()
            
    return img_urls
        
def popular(start_date, end_date):
    popular_articles = read_jsonl('popular_articles.jsonl')
    filtered_articles = [article for article in popular_articles if int(start_date) <= int(article['date']) <= int(end_date)]
    
    img_urls = popular_thread(filtered_articles)

    result = {
        'number_of_popular_articles': len(filtered_articles),
        'image_urls': img_urls,
    }

    with open(f'popular_{start_date}_{end_date}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def keyword_thread(filtered_articles, keyword):
    max_workers = os.cpu_count() or 1
    img_urls = []
    lock = Lock()

    def process_article(article, keyword):
        nonlocal img_urls
        soup = get_soup(article['url'])
        x = soup.body.text
        if not ('※ 發信站' in x and '作者' in x):
            return
        
        x = x.split('※ 發信站')[0]
        x = '作者' + x.split('作者')[1]
        
        if keyword not in x:
            return
        
        urls = [x for a in soup.find_all('a') if (x := a.get('href'))]
        urls = [url for url in urls if check_valid_pic_url(url)]
        
        with lock:
            img_urls.extend(urls)
        
        sleep(0.5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_article, article, keyword) for article in filtered_articles]
        
        for future in tqdm(futures):
            future.result()
            
    return img_urls
        
def keyword(start_date, end_date, keyword):
    articles = read_jsonl('articles.jsonl')
    filtered_articles = [article for article in articles if int(start_date) <= int(article['date']) <= int(end_date)]
    
    img_urls = keyword_thread(filtered_articles, keyword)

    result = {
        'image_urls': img_urls,
    }

    with open(f'keyword_{start_date}_{end_date}_{keyword}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    if len(sys.argv) < 2:
        print('Usage: python 110511233.py mode')
        return

    mode = sys.argv[1]
    
    # crawl, push, popular, keyword
    if mode == 'crawl':
        if len(sys.argv) != 2:
            print('Usage: python 110511233.py crawl')
            return
        crawl()
    elif mode == 'push':
        if len(sys.argv) != 4:
            print('Usage: python 110511233.py push start_date end_date')
            return
        push(sys.argv[2], sys.argv[3])
    elif mode == 'popular':
        if len(sys.argv) != 4:
            print('Usage: python 110511233.py popular start_date end_date')
            return
        popular(sys.argv[2], sys.argv[3])
    elif mode == 'keyword':
        if len(sys.argv) != 5:
            print('Usage: python 110511233.py keyword start_date end_date keyword')
            return
        keyword(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print('Invalid mode')
        
if __name__ == '__main__':
    main()