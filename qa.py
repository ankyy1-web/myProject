"""
qa.py
Interactive Q&A shell that imports main.py's pipeline
"""

import argparse
import time
from main import load_pipeline, ask_with_pipeline

def interactive_loop(pipeline, topk=5):
    print("🤖 PDF QA Agent Ready!")
    print("Type your questions (or 'exit' to quit).")
    while True:
        q = input("\nYour question: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break
        try:
            start_time = time.time()
            result, retrieved = ask_with_pipeline(pipeline, q, top_k=topk, return_retrieved=True)
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print("\n=== Answer ===")
        print(result.get("answer", "No answer generated."))

        citations = result.get("citations", [])
        if citations:
            print("\n=== Citations ===")
            for c in citations:
                print(f"- {c['filename']} p.{c['page']}")
        else:
            print("\n(No citations found)")

        if retrieved:
            print("\n=== Retrieved Chunks ===")
            for r in retrieved:
                fn = r['metadata'].get('filename'); pg = r['metadata'].get('page_number')
                txt = r['document'][:300].replace("\n", " ").strip()
                print(f"- {fn} p.{pg}: {txt}...")
        print(f"\n⏱ Response generated in {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small","base","large"], default="small", help="Model alias to use")
    parser.add_argument("--topk", type=int, default=5, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    print("⏳ Loading pipeline...")
    pipeline = load_pipeline(model_alias=args.model)
    print("✅ Pipeline ready!")
    interactive_loop(pipeline, topk=args.topk)
