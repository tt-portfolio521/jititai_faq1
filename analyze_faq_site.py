# -*- coding: utf-8 -*-
"""
汎用FAQ サイト分析スクリプト
各自治体のFAQサイト構造を自動解析し、カテゴリ・FAQ リンクをJSON出力する。
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import re
import time
import argparse
from collections import defaultdict
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
_BASE_DIR = Path(__file__).parent
OUTPUT_DIR = str(_BASE_DIR / "analysis_results")
MUNICIPALITIES_FILE = str(_BASE_DIR / "municipalities.json")


def fetch_page(url, timeout=30):
    """ページを取得してBeautifulSoupオブジェクトを返す"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        return BeautifulSoup(resp.text, "html.parser"), resp.url, resp.status_code
    except Exception as e:
        print(f"    取得エラー: {url} -> {e}")
        return None, url, 0


def find_faq_url(base_url):
    """ベースURLからFAQページの実際のURLを探索する"""
    soup, actual_url, status = fetch_page(base_url)
    if status == 200 and soup:
        return actual_url, soup

    # FAQ URLが見つからない場合、サイトトップから探す
    parsed = urlparse(base_url)
    site_root = f"{parsed.scheme}://{parsed.netloc}/"
    soup, _, status = fetch_page(site_root)
    if not soup:
        return base_url, None

    # FAQ/よくある質問 へのリンクを探す
    faq_patterns = ["faq", "qa", "よくある質問"]
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"].lower()
        if any(p in href or p in text for p in faq_patterns):
            full_url = urljoin(site_root, a["href"])
            faq_soup, actual, s = fetch_page(full_url)
            if s == 200:
                return actual, faq_soup

    return base_url, None


def extract_categories(soup, faq_url):
    """ページからカテゴリ構造を抽出する"""
    if not soup:
        return []

    parsed_faq = urlparse(faq_url)
    faq_domain = parsed_faq.netloc

    categories = []
    seen_urls = set()

    # 方法1: h2/h3 見出しの下のリンクをカテゴリとして認識
    for heading in soup.find_all(["h2", "h3"]):
        cat_name = heading.get_text(strip=True)
        if not cat_name or len(cat_name) > 30:
            continue

        # 見出しの後に続くリンクを探す
        sub_links = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h3"]:
            if sibling.name in ["ul", "ol", "div"]:
                for a in sibling.find_all("a", href=True):
                    href = a["href"]
                    full_url = urljoin(faq_url, href)
                    text = a.get_text(strip=True)
                    if (text and len(text) < 50 and full_url not in seen_urls
                            and faq_domain in full_url):
                        seen_urls.add(full_url)
                        sub_links.append({"name": text, "url": full_url})
            sibling = sibling.find_next_sibling() if sibling else None

        if sub_links:
            categories.append({
                "name": cat_name,
                "subcategories": sub_links
            })

    # 方法2: 見出しベースで見つからなかった場合、リンクパターンで推測
    if not categories:
        link_groups = defaultdict(list)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(faq_url, href)
            text = a.get_text(strip=True)

            if not text or len(text) > 80 or full_url in seen_urls:
                continue
            if faq_domain not in full_url:
                continue

            # FAQ関連のリンクを分類
            path = urlparse(full_url).path
            if any(p in path for p in ["/faq/", "/qa/", "/question/"]):
                # パスの一部をグループキーとして使用
                parts = path.strip("/").split("/")
                if len(parts) >= 2:
                    group_key = parts[1] if parts[0] in ("faq", "qa") else parts[0]
                else:
                    group_key = "general"

                seen_urls.add(full_url)
                link_groups[group_key].append({
                    "name": text,
                    "url": full_url,
                    "is_detail": len(text) > 15  # 長いテキスト→詳細ページ
                })

        # グループをカテゴリに変換
        for group_key, links in link_groups.items():
            cat_links = [l for l in links if not l["is_detail"]]
            detail_links = [l for l in links if l["is_detail"]]

            if cat_links:
                categories.append({
                    "name": group_key,
                    "subcategories": [{"name": l["name"], "url": l["url"]} for l in cat_links]
                })
            elif detail_links:
                categories.append({
                    "name": group_key,
                    "faq_links": [{"title": l["name"], "url": l["url"]} for l in detail_links[:5]]
                })

    # 方法3: 全リンクからFAQ個別ページリンクを直接抽出
    all_faq_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(faq_url, href)
        text = a.get_text(strip=True)

        if (text and 10 < len(text) < 100
                and faq_domain in full_url
                and full_url not in seen_urls):
            # FAQ質問っぽいリンクを判定（疑問形、長めのテキスト）
            if any(q in text for q in ["？", "?", "ですか", "ますか", "ません", "について", "方法", "手続", "届出"]):
                seen_urls.add(full_url)
                all_faq_links.append({"title": text, "url": full_url})

    return categories, all_faq_links


def get_sample_faq(url):
    """個別FAQページにアクセスしてQ&Aの構造パターンを確認する"""
    soup, _, status = fetch_page(url)
    if not soup or status != 200:
        return None

    result = {"url": url}

    # 質問タイトル: h1 を探す
    h1 = soup.find("h1")
    if h1:
        result["question"] = h1.get_text(strip=True)

    # 回答: 主要コンテンツエリアを探す
    content_selectors = [
        ("div", {"id": "js-article-body"}),
        ("div", {"class_": "article-body"}),
        ("div", {"class_": "contentGpArticleDoc"}),
        ("div", {"id": "tmp_contents"}),
        ("div", {"class_": "main_naka_kiji"}),
        ("article", {}),
        ("main", {}),
    ]

    for tag, attrs in content_selectors:
        elem = soup.find(tag, **attrs)
        if elem:
            text = elem.get_text(separator="\n", strip=True)
            if text and len(text) > 20:
                result["answer_preview"] = text[:300]
                result["answer_selector"] = f"{tag}#{attrs.get('id', attrs.get('class_', ''))}"
                break

    return result


def analyze_municipality(city_info):
    """1つの自治体のFAQサイトを分析する"""
    name = city_info["name"]
    faq_url = city_info["faq_url"]

    print(f"\n{'='*50}")
    print(f"■ 分析中: {name}")
    print(f"  URL: {faq_url}")
    print(f"{'='*50}")

    # FAQ URLの検証・探索
    actual_url, soup = find_faq_url(faq_url)
    if actual_url != faq_url:
        print(f"  リダイレクト: {actual_url}")

    if not soup:
        print(f"  ⚠ ページ取得失敗")
        return {
            "name": name,
            "faq_url": faq_url,
            "actual_url": actual_url,
            "status": "error",
            "error": "ページ取得失敗"
        }

    # カテゴリ構造を抽出
    categories, direct_faq_links = extract_categories(soup, actual_url)
    print(f"  カテゴリ: {len(categories)}件")
    print(f"  直接FAQリンク: {len(direct_faq_links)}件")

    # カテゴリの詳細表示
    for cat in categories[:10]:
        sub_count = len(cat.get("subcategories", []))
        faq_count = len(cat.get("faq_links", []))
        print(f"    - {cat['name']}: サブカテゴリ{sub_count}件, FAQ{faq_count}件")

    # サンプルFAQの取得（最大3件）
    sample_urls = []
    for cat in categories:
        for sub in cat.get("subcategories", [])[:1]:
            sample_urls.append(sub["url"])
    for faq in direct_faq_links[:2]:
        sample_urls.append(faq["url"])

    samples = []
    for url in sample_urls[:3]:
        time.sleep(1)
        sample = get_sample_faq(url)
        if sample:
            samples.append(sample)

    result = {
        "name": name,
        "faq_url": faq_url,
        "actual_url": actual_url,
        "status": "analyzed",
        "categories": categories,
        "direct_faq_links": direct_faq_links[:20],
        "total_direct_faq_links": len(direct_faq_links),
        "sample_faqs": samples,
    }

    # 結果をJSONファイルに保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  保存: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="FAQ サイト汎用分析スクリプト")
    parser.add_argument("--city", help="特定の自治体名を指定")
    parser.add_argument("--analyze-all", action="store_true", help="全pending自治体を分析")
    parser.add_argument("--status", action="store_true", help="処理状況を表示")
    args = parser.parse_args()

    # municipalities.json を読み込み
    with open(MUNICIPALITIES_FILE, "r", encoding="utf-8") as f:
        municipalities = json.load(f)

    if args.status:
        done = [m for m in municipalities if m["status"] == "done"]
        pending = [m for m in municipalities if m["status"] == "pending"]
        print(f"取得済み: {len(done)}市")
        print(f"未処理: {len(pending)}市")
        print(f"合計: {len(municipalities)}市")
        for m in pending:
            analyzed = os.path.exists(os.path.join(OUTPUT_DIR, f"{m['name']}.json"))
            status = "分析済" if analyzed else "未分析"
            print(f"  [{status}] {m['name']} ({m['size']}) - {m['region']}")
        return

    if args.city:
        # 特定の自治体を分析
        city = next((m for m in municipalities if m["name"] == args.city), None)
        if not city:
            print(f"エラー: '{args.city}' が見つかりません")
            return
        analyze_municipality(city)

    elif args.analyze_all:
        # 全自治体を分析
        to_process = municipalities
        print(f"分析対象: {len(to_process)}市")
        for i, city in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}]")
            analyze_municipality(city)
            time.sleep(2)

    else:
        # 引数なし → 全自治体を分析（--analyze-all と同じ）
        to_process = municipalities
        print(f"分析対象: {len(to_process)}市")
        for i, city in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}]")
            analyze_municipality(city)
            time.sleep(2)


if __name__ == "__main__":
    main()