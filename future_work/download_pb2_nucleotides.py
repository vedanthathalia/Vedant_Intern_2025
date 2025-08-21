import requests
import time
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm

search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
search_params = {
    "db": "nucleotide",
    "term": "Influenza A PB2[Title] AND segment 1[Title]",
    "retmax": 1000,
    "retmode": "json"
}

response = requests.get(search_url, params=search_params)
id_list = response.json()["esearchresult"]["idlist"]
print(f"Found {len(id_list)} PB2 nucleotide records")

fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
records = []

for nucleotide_id in tqdm(id_list, desc="Fetching records", unit="seq"):
    fetch_params = {
        "db": "nucleotide",
        "id": nucleotide_id,
        "retmode": "xml"
    }

    try:
        response = requests.get(fetch_url, params=fetch_params)
        root = ET.fromstring(response.content)
        gbseq = root.find(".//GBSeq")

        accession = gbseq.findtext("GBSeq_primary-accession", default="N/A")
        sequence = gbseq.findtext("GBSeq_sequence", default="").upper()

        host = segment = subtype = country = collection_date = "N/A"

        for feature in gbseq.findall(".//GBFeature"):
            if feature.findtext("GBFeature_key") == "source":
                for qual in feature.findall(".//GBQualifier"):
                    name = qual.findtext("GBQualifier_name")
                    value = qual.findtext("GBQualifier_value")
                    if name == "host":
                        host = value
                    elif name == "segment":
                        segment = value
                    elif name == "country":
                        country = value
                    elif name == "collection_date":
                        collection_date = value
                    elif name == "subtype":
                        subtype = value

        if not sequence:
            print(f"Skipping {accession}, sequence not found.")
            continue

        host_lower = host.lower()
        if "homo sapiens" in host_lower or "human" in host_lower:
            label_human, label_avian, label_other = 1, 0, 0
        elif any(bird in host_lower for bird in ["duck", "chicken", "avian", "goose", "mallard", "turkey", "quail"]):
            label_human, label_avian, label_other = 0, 1, 0
        else:
            label_human, label_avian, label_other = 0, 0, 1

        records.append({
            "accession": accession,
            "sequence": sequence,
            "host": host,
            "segment": segment,
            "subtype": subtype,
            "country": country,
            "collection_date": collection_date,
            "label_human": label_human,
            "label_avian": label_avian,
            "label_other": label_other
        })

        time.sleep(0.34)

    except Exception as e:
        print(f"Error processing ID {nucleotide_id}: {e}")
        continue

csv_file = "pb2_nucleotide_sequences.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"Saved {len(records)} PB2 records to {csv_file}")
