import json                                                                        
from pathlib import Path                                                           
from huggingface_hub import hf_hub_download                                        

# Download the raw data files                                                      
print("Downloading data files...")                                                 
faq_file = hf_hub_download(repo_id="qgyd2021/e_commerce_customer_service",         
filename="data/faq.json", repo_type="dataset")                                     
product_file = hf_hub_download(repo_id="qgyd2021/e_commerce_customer_service",     
filename="data/product.jsonl", repo_type="dataset")                                

# Load FAQ data                                                                    
with open(faq_file, 'r', encoding='utf-8') as f:
    faq_data     = json.load(f)                                                        

# Load product data (JSONL)                                                        
product_data = []                                                                  
with open(product_file, 'r', encoding='utf-8') as f:
    for line in f:                                                                 
        if line.strip():                                                           
            product_data.append(json.loads(line))                                  

print(f"\n✓ Loaded {len(faq_data)} FAQ entries")                                   
print(f"✓ Loaded {len(product_data)} product entries")                             
print("\n=== FAQ Sample ===")                                                      
print(json.dumps(faq_data[0], indent=2, ensure_ascii=False)[:500])                 
print("\n\n=== Product Sample ===")                                                
print(json.dumps(product_data[0], indent=2, ensure_ascii=False)[:500])             

# Convert to SQuAD-like format                                                        
paragraphs = []
for idx, faq in enumerate(faq_data):
    context = faq.get("answer", "")
    qas = [
        {
            "id": f"faq-{idx:04d}",
            "question": faq.get("question", ""),
            "answers": [
                {
                    "text": context,
                    "answer_start": 0,
                }
            ],
        }
    ]
    paragraphs.append({"context": context, "qas": qas})

squad_payload = {
    "version": "1.0",
    "data": [
        {
            "title": "ecommerce_faqs",
            "paragraphs": paragraphs,
        }
    ],
}

output_file = Path(__file__).resolve().parent.parent.parent / "data" / "ecommerce_faq_as_squad.json"
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w", encoding="utf-8") as f:
    json.dump(squad_payload, f, ensure_ascii=False, indent=2)

print(f"\nConverted FAQ data to SQuAD JSON at {output_file}")