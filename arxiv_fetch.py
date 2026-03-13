import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_arxiv_metadata(arxiv_id: str):
  try:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
      print(f"⚠️ arXiv API error for {arxiv_id}: {resp.status_code}")
      return None

    root = ET.fromstring(resp.text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    entry = root.find("atom:entry", ns)
    if entry is None:
      return None

    title = entry.findtext("atom:title", default="", namespaces=ns).strip()
    summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
    published = entry.findtext("atom:published", default="", namespaces=ns)
    updated = entry.findtext("atom:updated", default="", namespaces=ns)
    authors = [
        a.findtext("atom:name", default="", namespaces=ns)
        for a in entry.findall("atom:author", ns)
    ]
    categories = [
        c.attrib.get("term") for c in entry.findall("atom:category", ns)
    ]
    primary_cat = entry.find("arxiv:primary_category", ns)
    primary_term = primary_cat.attrib.get(
        "term") if primary_cat is not None else None
    journal_ref = entry.findtext("arxiv:journal_ref",
                                 default="",
                                 namespaces=ns)
    doi = entry.findtext("arxiv:doi", default="", namespaces=ns)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "authors": authors,
        "published": published,
        "updated": updated,
        "categories": categories,
        "primary_category": primary_term,
        "journal_ref": journal_ref,
        "doi": doi,
    }
  except Exception as e:
    print(f"⚠️ Failed to fetch arXiv metadata for {arxiv_id}: {e}")
    return None


def fetch_arxiv_batch(arxiv_ids, max_workers: int = 8):
  results = []
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_id = {
        executor.submit(fetch_arxiv_metadata, arxiv_id): arxiv_id
        for arxiv_id in arxiv_ids
    }
    for future in as_completed(future_to_id):
      arxiv_id = future_to_id[future]
      try:
        data = future.result()
        if data:
          results.append(data)
          print(f"✅ Metadata fetched for {arxiv_id}")
        else:
          print(f"⚠️ No data for {arxiv_id}")
      except Exception as e:
        print(f"❌ Error processing {arxiv_id}: {e}")
  return results
