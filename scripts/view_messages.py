#!/usr/bin/env python3
"""
Simple script to view the enhanced realistic messages in a readable format.
"""

import json
import sys
from collections import defaultdict


def view_messages(episode_file: str):
    """View messages from an episode file in a readable format."""

    print(f"Viewing messages from: {episode_file}")
    print("=" * 80)

    # Load episode data
    steps = []
    with open(episode_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("type") == "step":
                steps.append(data)

    # Group messages by conversation thread
    threads = defaultdict(list)

    for step in steps:
        for message in step.get("messages", []):
            thread_id = message.get("thread_id", "no_thread")
            threads[thread_id].append(
                {
                    "step": step["step"],
                    "sender_id": message["sender_id"],
                    "receiver_id": message.get(
                        "receiver_id", -1
                    ),  # Default to -1 (broadcast) if not present
                    "agent_type": message["agent_type"],
                    "message": message["message"],
                    "message_type": message.get("message_type", "unknown"),
                    "is_response": message.get("is_response", False),
                }
            )

    print(f"Found {len(threads)} conversation threads")
    print(f"Total messages: {sum(len(thread) for thread in threads.values())}")
    print()

    # Display conversation threads
    for i, (thread_id, messages) in enumerate(threads.items(), 1):
        print(f"CONVERSATION THREAD {i} ({thread_id})")
        print("-" * 60)

        for msg in messages:
            sender = f"Agent {msg['sender_id']} ({msg['agent_type']})"
            receiver = (
                f"Agent {msg['receiver_id']}" if msg["receiver_id"] != -1 else "All"
            )
            response_indicator = " [RESPONSE]" if msg["is_response"] else ""

            print(f"Step {msg['step']:2d}: {sender} → {receiver}{response_indicator}")
            print(f"         {msg['message']}")
            print()

        print()

    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 40)

    total_messages = sum(len(thread) for thread in threads.values())
    responses = sum(
        1 for thread in threads.values() for msg in thread if msg["is_response"]
    )

    print(f"Total messages: {total_messages}")
    print(f"Responses: {responses}")
    print(f"Response rate: {(responses/total_messages)*100:.1f}%")
    print(f"Average thread length: {total_messages/len(threads):.1f} messages")

    # Message type breakdown
    message_types = defaultdict(int)
    for thread in threads.values():
        for msg in thread:
            message_types[msg["message_type"]] += 1

    print("\nMessage types:")
    for msg_type, count in message_types.items():
        print(f"  {msg_type}: {count}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        episode_file = sys.argv[1]
    else:
        episode_file = "logs/enhanced_realistic_episode_20250921_002645.jsonl"

    view_messages(episode_file)
