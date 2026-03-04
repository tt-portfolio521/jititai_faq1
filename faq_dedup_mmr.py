"""
自治体FAQデータ統合 → MMRによる代表質問選出スクリプト（Gemini Embedding API版）

処理フロー:
  ① 指定フォルダ内の全Excelから質問・回答をカテゴリ別に抽出・統合
  ② 質問テキストの正規化・自治体固有質問の除外
  ③ Gemini Embedding APIで全質問をベクトル化（非同期並列処理）
  ④ カテゴリごとにMMRで代表質問を選出
  → 出力: カテゴリ別シートのExcelファイル

使い方:
  export GEMINI_API_KEY="your-api-key"
  python faq_dedup_mmr.py --input ./faq_data --output ./代表質問一覧.xlsx

  # パラメータ調整例
  python faq_dedup_mmr.py -i ./faq_data -o ./output.xlsx --lambda 0.3 --max-ratio 0.5

必要パッケージ:
  pip install pandas openpyxl numpy scikit-learn aiohttp
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import glob
import asyncio
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 設定（デフォルト値）
# ============================================================
GEMINI_MODEL = "gemini-embedding-001"

# MMR パラメータ
DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_MMR_SCORE_THRESHOLD = -0.5
DEFAULT_MMR_MAX_RATIO = 0.3

# Gemini API 並列処理パラメータ
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_RATE_LIMIT_DELAY = 0.5


# ============================================================
# 自治体固有質問フィルタ
# ============================================================
# 自治体名リスト（ファイル名から自動的に追加される + 手動追加分）
EXTRA_CITY_NAMES = {
    # --- 大都市 ---
    "札幌市", "仙台市", "さいたま市", "千葉市", "横浜市",
    "川崎市", "新潟市", "静岡市", "浜松市", "名古屋市",
    "京都市", "大阪市", "神戸市", "広島市", "岡山市",
    "北九州市", "福岡市",
    # --- 中都市 ---
    "旭川市", "盛岡市", "秋田市", "郡山市", "宇都宮市",
    "川越市", "柏市", "八王子市", "金沢市", "長野市",
    "豊橋市", "奈良市", "明石市", "倉敷市", "松山市",
    "高松市", "大分市",
    # --- 小都市 ---
    "函館市", "弘前市", "鶴岡市", "日立市", "鎌倉市",
    "小田原市", "上田市", "津市", "敦賀市", "生駒市", "米子市",
    "呉市", "四万十市", "佐世保市", "延岡市", "那覇市",
    # --- その他 ---
    "堺市", "熊本市", "相模原市",
}


def is_city_specific(question: str, city_name: str) -> bool:
    """質問が特定の自治体固有の内容かどうかを判定"""
    q = question.strip()
    all_city_names = EXTRA_CITY_NAMES | {city_name}

    # 自治体名（○○市）が含まれていれば除外
    for cn in all_city_names:
        if cn in q:
            return True

    # 「市」なしの駅名パターン（「札幌駅」「仙台駅」等）
    for cn in all_city_names:
        base = re.sub(r'[市町村区]$', '', cn)
        if base and len(base) >= 2 and base + "駅" in q:
            return True

    # 広報紙名（「広報かまくら」「広報さっぽろ」等）
    if re.search(r'広報[ぁ-んァ-ン一-鿿]{2,}', q):
        return True
    # 「市政だより」「市民の友」等の固有広報紙名
    if re.search(r'(市政だより|市民の友|掲示板)', q):
        return True

    return False


def strip_city_names(text: str, city_name: str = "") -> str:
    """質問文から自治体名を除去してEmbedding精度を向上させる。

    例:
      "広報さいたま市が届かない" → "広報が届かない"
      "さいたま市のがん検診の費用" → "がん検診の費用"
    """
    all_names = EXTRA_CITY_NAMES | ({city_name} if city_name else set())
    for cn in sorted(all_names, key=len, reverse=True):
        if cn and cn in text:
            text = text.replace(cn, "")
    # 広報紙の固有名も汎用化（「広報かまくら」→「広報」）
    text = re.sub(r'(広報)[ぁ-んァ-ン]{2,}', r'\1', text)
    text = re.sub(r'(広報紙)[「」ぁ-んァ-ン]+[「」]?', r'\1', text)
    text = re.sub(r'^[のはがをにへでと、]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# 統一カテゴリ定義（LLM分類の基準となる20カテゴリ + その他）
# ============================================================
UNIFIED_CATEGORIES = [
    "住民票・戸籍・届出", "健康・医療", "子育て・教育", "税金",
    "環境・ごみ", "福祉・介護", "防災・消防", "保険・年金",
    "住まい・生活", "産業・労働", "市政・行政", "選挙", "相談",
    "施設", "参画・交流", "入札・契約", "観光・文化", "交通",
    "人権", "マイナンバー", "その他"
]

# LLM分類結果のキャッシュ（同じシート名の再分類を防止してAPI節約）
_sheet_category_cache: dict[str, str] = {}


def classify_sheet_with_gemini(sheet_name: str, sample_questions: list[str],
                                api_key: str) -> str:
    """Gemini APIを使ってシート名＋質問サンプルから統一カテゴリを自動判定する。

    Args:
        sheet_name: Excelのシート名
        sample_questions: そのシートに含まれる質問のサンプル（3〜5件）
        api_key: Gemini API キー

    Returns:
        UNIFIED_CATEGORIES のいずれか1つ
    """
    # キャッシュヒット → API呼び出し不要
    if sheet_name in _sheet_category_cache:
        cached = _sheet_category_cache[sheet_name]
        print(f"  [CACHE] '{sheet_name}' → '{cached}'")
        return cached

    # カテゴリ一覧文字列
    categories_str = "\n".join(f"- {c}" for c in UNIFIED_CATEGORIES)

    # 質問サンプル文字列（最大5件）
    samples_str = ""
    if sample_questions:
        samples = sample_questions[:5]
        samples_str = "\n".join(f"  - {q}" for q in samples)
        samples_str = f"\n\n含まれる質問の例:\n{samples_str}"

    prompt = f"""あなたは自治体FAQのカテゴリ分類の専門家です。
以下のシート名（と質問サンプル）を見て、最も適切なカテゴリを1つだけ出力してください。

シート名: 「{sheet_name}」{samples_str}

カテゴリ一覧:
{categories_str}

【ルール】
- 上記カテゴリ一覧の中から1つだけ選んでください
- カテゴリ名のみを出力してください（説明や記号は不要）
- どれにも当てはまらない場合は「その他」と出力してください"""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )

        result = response.text.strip().strip("「」『』\"' ")

        # LLMの応答が定義済みカテゴリに含まれるか検証
        if result not in UNIFIED_CATEGORIES:
            # 部分一致でフォールバック
            for cat in UNIFIED_CATEGORIES:
                if cat in result or result in cat:
                    result = cat
                    break
            else:
                print(f"  [WARN] LLM応答 '{result}' は未知カテゴリ → 'その他'")
                result = "その他"

        _sheet_category_cache[sheet_name] = result
        print(f"  [LLM] '{sheet_name}' → '{result}'")
        return result

    except Exception as e:
        print(f"  [ERROR] LLM分類失敗 '{sheet_name}': {e} → 'その他'")
        _sheet_category_cache[sheet_name] = "その他"
        return "その他"


# ============================================================
# ① データ読み込み・統合
# ============================================================
# 読み込みから除外するファイルパターン（代表質問一覧=出力ファイル）
EXCLUDE_FILE_PATTERNS = [
    "代表質問一覧",  # 前回の出力ファイル
]


def load_all_faq(input_dir: str, api_key: str) -> pd.DataFrame:
    all_xlsx = sorted(set(glob.glob(os.path.join(input_dir, "*.xlsx"))))
    # 除外パターンに該当するファイルを除外
    files = []
    excluded = []
    for f in all_xlsx:
        fname = Path(f).name
        if any(pat in fname for pat in EXCLUDE_FILE_PATTERNS):
            excluded.append(fname)
        else:
            files.append(f)

    if not files:
        raise FileNotFoundError(f"Excelファイルが見つかりません: {input_dir}")

    print(f"検出ファイル: {len(files)}件")
    for f in files:
        print(f"  - {Path(f).name}")
    if excluded:
        print(f"除外ファイル: {len(excluded)}件")
        for e in excluded:
            print(f"  ✕ {e}")

    all_rows = []
    skipped_no_question = 0
    skipped_city_specific = 0

    for fpath in files:
        city_name = re.sub(r'FAQ.*$', '', Path(fpath).stem).strip() or Path(fpath).stem
        xl = pd.ExcelFile(fpath)

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name=sheet_name)
            if "質問" not in df.columns or "回答" not in df.columns:
                print(f"  [SKIP] {city_name} / {sheet_name}: '質問'または'回答'列なし")
                continue

            # 質問サンプルを取得してLLMに渡す（最大5件）
            sample_qs = []
            if "質問" in df.columns:
                sample_qs = df["質問"].dropna().head(5).tolist()

            unified_category = classify_sheet_with_gemini(sheet_name, sample_qs, api_key)

            for _, row in df.iterrows():
                q = str(row.get("質問", "")).strip()
                a = str(row.get("回答", "")).strip()
                sub = str(row.get("サブカテゴリ", "")).strip()

                if not q or q == "nan" or q == "キーワードで検索":
                    skipped_no_question += 1
                    continue

                if is_city_specific(q, city_name):
                    skipped_city_specific += 1
                    continue

                all_rows.append({
                    "自治体": city_name,
                    "元カテゴリ": sheet_name,
                    "統一カテゴリ": unified_category,
                    "サブカテゴリ": sub if sub != "nan" else "",
                    "質問": q,
                    "回答": a,
                })

    result = pd.DataFrame(all_rows)
    print(f"\n読み込み完了: {len(files)}ファイル, {len(result)}件")
    if skipped_no_question:
        print(f"  無効な質問でスキップ: {skipped_no_question}件")
    if skipped_city_specific:
        print(f"  自治体固有質問で除外: {skipped_city_specific}件")

    print("\nカテゴリ別件数:")
    for cat, cnt in result["統一カテゴリ"].value_counts().items():
        print(f"  {cat}: {cnt}件")

    unmapped = result[result["統一カテゴリ"] == "その他"]["元カテゴリ"].unique()
    if len(unmapped) > 0:
        print(f"\n⚠ 'その他'に分類されたシート名:")
        for s in unmapped:
            print(f"    - '{s}'")

    return result


# ============================================================
# ② テキスト正規化
# ============================================================
def clean_question(text: str) -> str:
    text = re.sub(r'[　\s]+', ' ', text).strip()
    text = re.sub(r'^Q[\.:：\s]*', '', text)
    text = re.sub(r'[？?]+$', '', text).strip()
    return text


def strip_city_names(text: str, city_name: str = "") -> str:
    """質問文から自治体名を除去してEmbedding精度を向上させる。

    例:
      "広報さいたま市が届かない" → "広報が届かない"
      "さいたま市のがん検診の費用と受け方" → "がん検診の費用と受け方"
    """
    all_names = EXTRA_CITY_NAMES | ({city_name} if city_name else set())
    for cn in sorted(all_names, key=len, reverse=True):
        if cn and cn in text:
            text = text.replace(cn, "")
    text = re.sub(r'^[のはがをにへでと、]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_answer(text: str) -> str:
    text = re.sub(r'\$\(function\(\)\s*\{.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[　\s]+', ' ', text).strip()
    return text


# ============================================================
# ③ Gemini Embedding API（非同期並列処理）
# ============================================================
def embed_with_gemini(texts: list[str], api_key: str,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                      rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY) -> np.ndarray:
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp が必要です。  pip install aiohttp")
        sys.exit(1)

    async def _run():
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(batches)
        print(f"\nGemini Embedding: {len(texts)}件 → {total}バッチ "
              f"(バッチサイズ{batch_size}, 並列数{max_concurrent})")

        async def process_batch(batch_texts, session, batch_id):
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{GEMINI_MODEL}:batchEmbedContents?key={api_key}")
            body = {"requests": [
                {"model": f"models/{GEMINI_MODEL}",
                 "content": {"parts": [{"text": t}]},
                 "taskType": "CLUSTERING",
                 "outputDimensionality": 1536}
                for t in batch_texts
            ]}

            for attempt in range(3):
                async with semaphore:
                    try:
                        async with session.post(url, json=body) as resp:
                            if resp.status == 429:
                                wait = float(resp.headers.get("Retry-After", 5))
                                print(f"  [RATE LIMIT] バッチ{batch_id}: "
                                      f"{wait}秒待機 (リトライ{attempt+1}/3)")
                                await asyncio.sleep(wait)
                                continue
                            elif resp.status != 200:
                                error = await resp.text()
                                print(f"  [ERROR] バッチ{batch_id}: "
                                      f"status={resp.status}, {error[:200]}")
                                return [None] * len(batch_texts)
                            data = await resp.json()
                            return [e.get("values")
                                    for e in data.get("embeddings", [])]
                    except Exception as e:
                        print(f"  [ERROR] バッチ{batch_id}: {e}")
                        if attempt < 2:
                            await asyncio.sleep(2)
                    finally:
                        await asyncio.sleep(rate_limit_delay)
            return [None] * len(batch_texts)

        async with aiohttp.ClientSession() as session:
            tasks = [process_batch(b, session, i)
                     for i, b in enumerate(batches)]
            results = await asyncio.gather(*tasks)

        all_embs, failed, dim = [], 0, None
        for batch_result in results:
            for emb in batch_result:
                if emb is None:
                    failed += 1
                    all_embs.append(None)
                else:
                    if dim is None:
                        dim = len(emb)
                    all_embs.append(np.array(emb))

        if dim is None:
            raise RuntimeError(
                "全Embeddingに失敗しました。"
                "APIキーとネットワーク接続を確認してください。")

        for i in range(len(all_embs)):
            if all_embs[i] is None:
                all_embs[i] = np.zeros(dim)

        if failed:
            print(f"  ⚠ {failed}件のEmbeddingに失敗（ゼロベクトルで代替）")

        print(f"Embedding完了: {len(all_embs)}件, 次元数: {dim}")
        return np.array(all_embs)

    return asyncio.run(_run())


# ============================================================
# ④ MMR（Maximal Marginal Relevance）
# ============================================================
def mmr_select(embeddings, indices,
               lam=DEFAULT_MMR_LAMBDA,
               score_threshold=DEFAULT_MMR_SCORE_THRESHOLD,
               max_ratio=DEFAULT_MMR_MAX_RATIO):
    """MMR選出（ベクトル化高速版）"""
    if len(indices) <= 1:
        return indices

    sub = embeddings[indices]
    n = len(indices)
    max_select = max(1, int(n * max_ratio))

    # 事前に全ペアの類似度行列を計算（ボトルネック解消）
    sim_matrix = cosine_similarity(sub)  # (n, n)

    centroid = sub.mean(axis=0, keepdims=True)
    relevance = cosine_similarity(sub, centroid).flatten()  # (n,)

    selected = []
    remaining = set(range(n))

    # 最も関連性の高いものを最初に選択
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.discard(first)

    # 各候補の「選択済みとの最大類似度」を追跡
    max_sim_to_selected = sim_matrix[:, first].copy()  # (n,)

    remaining_arr = np.array(list(remaining))

    while len(remaining) > 0 and len(selected) < max_select:
        # ベクトル化でMMRスコアを一括計算
        r_idx = remaining_arr
        scores = lam * relevance[r_idx] - (1 - lam) * max_sim_to_selected[r_idx]

        best_pos = int(np.argmax(scores))
        best_score = scores[best_pos]

        if best_score < score_threshold:
            break

        best_idx = r_idx[best_pos]
        selected.append(int(best_idx))
        remaining.discard(int(best_idx))

        # remaining_arrを更新（選択されたものを除去）
        remaining_arr = np.delete(remaining_arr, best_pos)

        # 新しく選択されたものとの類似度でmax_sim_to_selectedを更新
        np.maximum(max_sim_to_selected, sim_matrix[:, best_idx], out=max_sim_to_selected)

    return [indices[i] for i in selected]


# ============================================================
# Excel出力
# ============================================================
def save_to_excel(df, category_stats, output_path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    wb.remove(wb.active)

    hf = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    hfill = PatternFill("solid", fgColor="4472C4")
    ha = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cf = Font(name="Arial", size=10)
    ca = Alignment(vertical="top", wrap_text=True)
    tb = Border(left=Side("thin"), right=Side("thin"),
                top=Side("thin"), bottom=Side("thin"))

    # --- サマリーシート ---
    ws = wb.create_sheet("サマリー", 0)
    for ci, h in enumerate(["カテゴリ", "元の質問数", "代表質問数", "削減率"], 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font, c.fill, c.alignment, c.border = hf, hfill, ha, tb

    for ri, cat in enumerate(sorted(category_stats.keys()), 2):
        s = category_stats[cat]
        red = f"{(1-s['after']/s['before'])*100:.0f}%" if s['before'] > 0 else "-"
        for ci, v in enumerate([cat, s["before"], s["after"], red], 1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.font, c.border = cf, tb

    tr = len(category_stats) + 2
    tb_before = sum(s["before"] for s in category_stats.values())
    tb_after = sum(s["after"] for s in category_stats.values())
    red_t = f"{(1-tb_after/tb_before)*100:.0f}%" if tb_before > 0 else "-"
    bf = Font(name="Arial", bold=True, size=11)
    for ci, v in enumerate(["合計", tb_before, tb_after, red_t], 1):
        c = ws.cell(row=tr, column=ci, value=v)
        c.font, c.border = bf, tb

    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 15
    ws.column_dimensions["C"].width = 15
    ws.column_dimensions["D"].width = 12

    # --- カテゴリ別シート ---
    headers = ["No.", "統一カテゴリ", "サブカテゴリ",
               "質問", "回答（参考）", "出典自治体"]
    for cat in sorted(df["統一カテゴリ"].unique()):
        cdf = df[df["統一カテゴリ"] == cat].reset_index(drop=True)
        sn = re.sub(r'[\\/*?\[\]:]', '', cat)[:31]
        ws = wb.create_sheet(title=sn)

        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=ci, value=h)
            c.font, c.fill, c.alignment, c.border = hf, hfill, ha, tb

        for ri, (_, row) in enumerate(cdf.iterrows(), 2):
            ans = str(row["回答"])
            if len(ans) > 500:
                ans = ans[:500] + "..."
            vals = [ri-1, row["統一カテゴリ"], row.get("サブカテゴリ", ""),
                    row["質問"], ans, row["自治体"]]
            for ci, v in enumerate(vals, 1):
                c = ws.cell(row=ri, column=ci, value=v)
                c.font, c.alignment, c.border = cf, ca, tb

        for ci, w in enumerate([6, 18, 18, 50, 60, 12], 1):
            ws.column_dimensions[ws.cell(1, ci).column_letter].width = w
        ws.auto_filter.ref = f"A1:F{len(cdf)+1}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wb.save(output_path)
    print(f"\n出力完了: {output_path}")


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="自治体FAQ 代表質問抽出ツール（Gemini API版）")
    parser.add_argument("--input", "-i", required=True,
                        help="FAQのExcelファイル格納ディレクトリ")
    parser.add_argument("--output", "-o", default="./代表質問一覧.xlsx",
                        help="出力Excelファイルパス")
    parser.add_argument("--api-key",
                        help="Gemini APIキー（環境変数 GEMINI_API_KEY でも可）")
    parser.add_argument("--lambda", dest="mmr_lambda", type=float,
                        default=DEFAULT_MMR_LAMBDA,
                        help=f"MMRのλ値 (default: {DEFAULT_MMR_LAMBDA})")
    parser.add_argument("--max-ratio", type=float, default=DEFAULT_MMR_MAX_RATIO,
                        help=f"カテゴリあたり最大選出比率 (default: {DEFAULT_MMR_MAX_RATIO})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MMR_SCORE_THRESHOLD,
                        help=f"MMR打ち切り閾値 (default: {DEFAULT_MMR_SCORE_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"APIバッチサイズ (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help=f"最大並列数 (default: {DEFAULT_MAX_CONCURRENT})")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Gemini APIキーが必要です。")
        print("  --api-key YOUR_KEY または $env:GEMINI_API_KEY='YOUR_KEY'")
        sys.exit(1)

    print("=" * 60)
    print("自治体FAQ 代表質問抽出ツール（Gemini API版）")
    print(f"  入力: {args.input}")
    print(f"  出力: {args.output}")
    print(f"  MMR: λ={args.mmr_lambda}, 最大比率={args.max_ratio}, "
          f"閾値={args.threshold}")
    print(f"  API: バッチ={args.batch_size}, 並列={args.max_concurrent}")
    print("=" * 60)

    # ① 読み込み
    print("\n【STEP 1】データ読み込み（自治体固有質問を自動除外）")
    df = load_all_faq(args.input, api_key)
    if len(df) == 0:
        print("有効な質問データがありません。")
        sys.exit(1)

    # ② 正規化
    print("\n【STEP 2】テキスト正規化")
    df["質問_clean"] = df["質問"].apply(clean_question)
    df["回答"] = df["回答"].apply(clean_answer)

    # Embedding用: 自治体名を除去（「広報○○市」→「広報」で統一）
    df["質問_embed"] = df.apply(
        lambda row: strip_city_names(row["質問_clean"], row.get("自治体", "")),
        axis=1,
    )
    n_changed = len(df[df["質問_clean"] != df["質問_embed"]])
    if n_changed > 0:
        print(f"  自治体名除去: {n_changed}件が変更")
        for _, row in df[df["質問_clean"] != df["質問_embed"]].head(3).iterrows():
            print(f"    {row['質問_clean'][:35]} → {row['質問_embed'][:35]}")

    before = len(df)
    df = df.drop_duplicates(subset=["質問_embed"], keep="first").reset_index(drop=True)
    print(f"  完全一致除去: {before} → {len(df)}件（{before-len(df)}件除去）")

    # ③ Embedding
    print("\n【STEP 3】Gemini Embedding（非同期並列処理）")
    embeddings = embed_with_gemini(
        df["質問_embed"].tolist(),
        api_key,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )

    # ④ MMR
    print("\n【STEP 4】MMRによる代表質問選出")
    stats, selected_indices = {}, []
    for cat in sorted(df["統一カテゴリ"].unique()):
        idxs = df[df["統一カテゴリ"] == cat].index.tolist()
        if not idxs:
            continue
        sel = mmr_select(embeddings, idxs,
                         lam=args.mmr_lambda,
                         score_threshold=args.threshold,
                         max_ratio=args.max_ratio)
        selected_indices.extend(sel)
        stats[cat] = {"before": len(idxs), "after": len(sel)}
        print(f"  {cat}: {len(idxs)}件 → {len(sel)}件")

    result = df.loc[selected_indices].reset_index(drop=True)
    result_emb = embeddings[selected_indices]
    total_before, total_after = len(df), len(result)
    print(f"\n合計: {total_before}件 → {total_after}件"
          f"（{total_after/total_before*100:.1f}%）")

    # ④-2 カテゴリ間の重複除去
    print("\n【STEP 4-2】カテゴリ間の重複質問を除去")
    sim = cosine_similarity(result_emb)
    np.fill_diagonal(sim, 0)

    removed = set()
    cross_dup_log = []
    CROSS_DEDUP_THRESHOLD = 0.95

    for i in range(len(result)):
        if i in removed:
            continue
        for j in range(i + 1, len(result)):
            if j in removed:
                continue
            # 同じカテゴリ内はMMRで処理済みなのでスキップ
            if result.iloc[i]["統一カテゴリ"] == result.iloc[j]["統一カテゴリ"]:
                continue
            if sim[i, j] >= CROSS_DEDUP_THRESHOLD:
                removed.add(j)
                cross_dup_log.append((
                    result.iloc[j]["質問"][:50],
                    result.iloc[j]["統一カテゴリ"],
                    result.iloc[i]["質問"][:50],
                    result.iloc[i]["統一カテゴリ"],
                    sim[i, j],
                ))

    if cross_dup_log:
        print(f"  {len(cross_dup_log)}件の重複を除去（閾値={CROSS_DEDUP_THRESHOLD}）:")
        for q_rm, cat_rm, q_kp, cat_kp, s in cross_dup_log[:10]:
            print(f"    除去: [{cat_rm}] {q_rm}")
            print(f"    残す: [{cat_kp}] {q_kp} (類似度={s:.3f})")
        if len(cross_dup_log) > 10:
            print(f"    ... 他{len(cross_dup_log) - 10}件")

        keep_mask = [i for i in range(len(result)) if i not in removed]
        result = result.iloc[keep_mask].reset_index(drop=True)

        # statsを更新
        for cat in stats:
            stats[cat]["after"] = len(result[result["統一カテゴリ"] == cat])

        print(f"  {total_after}件 → {len(result)}件")
        total_after = len(result)
    else:
        print("  カテゴリ間の重複なし")

    # ⑤ 出力
    print("\n【STEP 5】Excel出力")
    save_to_excel(result, stats, args.output)

    print("\n" + "=" * 60)
    print("処理完了！")
    print(f"  入力: {total_before}件 → 出力: {total_after}件")
    print(f"  ファイル: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()