import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.data_gen.agents import AuthorAgent, CriticAgent, ArchivistAgent

def deep_dive(topic, category="General", max_rounds=3):
    print(f"\n🚀 STARTING DEEP DIVE: {category} - {topic}")
    
    author = AuthorAgent()
    critic = CriticAgent()
    scribe = ArchivistAgent(output_root=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_base'))
    
    if scribe.exists(category, topic):
        print(f"⏩ Skipping '{topic}' (Already exists)")
        return
    
    current_draft = None
    feedback = None
    
    for i in range(max_rounds):
        round_name = f"Round {i+1}"
        print(f"\n--- {round_name} ---")
        
        # 1. Author drafts/refines
        current_draft = author.draft(topic, category=category, feedback=feedback, previous_content=current_draft)
        
        # 2. Critic reviews
        review_json = critic.review(topic, current_draft)
        try:
            review = json.loads(review_json)
            status = review.get("status", "FAIL")
            feedback = review.get("feedback", "No feedback provided.")
            
            print(f"[{status}] Critic Feedback: {feedback}")
            
            if status == "PASS":
                print("🏆 Critic Approved!")
                break
        except Exception as e:
            print(f"Review Parse Error: {e}")
            feedback = "Format error in review. Please check the draft again."
            
    # Final cleanup
    print(f"\n✅ Saving final version of '{topic}'...")
    scribe.archive(category, topic, current_draft)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = input("Enter topic to mine: ")
    
    deep_dive(topic)
