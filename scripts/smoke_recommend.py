import sys
from pathlib import Path
# Ensure project root is on path when executed via absolute path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from legal_xai.recommend import recommend_lawyers

samples = [
    "Need anticipatory bail for 498A domestic violence allegations",
    "Draft merger term sheet and shareholder rights compliance",
    "Article 14 equality violation during property acquisition",
    "Child custody and maintenance after mutual consent divorce",
    "Eviction notice for non payment of rent and injunction for trespass",
]

for s in samples:
    print("Query:", s)
    out = recommend_lawyers(s, top_k=3)
    print(out)
    print("-" * 60)
