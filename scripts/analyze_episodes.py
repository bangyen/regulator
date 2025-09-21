#!/usr/bin/env python3
"""
Comprehensive episode analysis utility.
Combines message analysis, conversation realism, and episode viewing functionality.
"""

import json
import sys
from typing import Dict, Any, Optional
from collections import defaultdict, Counter
import argparse


def load_episode_data(file_path: str) -> Dict[str, Any]:
    """Load episode data from JSONL file."""
    steps = []
    header = None
    summary = None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("type") == "episode_header":
                header = data
            elif data.get("type") == "step":
                steps.append(data)
            elif data.get("type") == "episode_summary":
                summary = data

    return {"header": header, "steps": steps, "summary": summary}


def analyze_messages(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze messages in the episode."""
    all_messages = []
    message_types = Counter()
    agent_messages = defaultdict(list)

    for step in episode_data["steps"]:
        if "messages" in step:
            for msg in step["messages"]:
                all_messages.append(msg)
                message_types[msg.get("type", "unknown")] += 1
                agent_messages[msg.get("sender_id", "unknown")].append(msg)

    return {
        "total_messages": len(all_messages),
        "message_types": dict(message_types),
        "agent_messages": dict(agent_messages),
        "all_messages": all_messages,
    }


def analyze_conversation_realism(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze conversation realism and patterns."""
    steps = episode_data["steps"]

    # Count messages per step
    messages_per_step = [len(step.get("messages", [])) for step in steps]

    # Analyze message types
    message_types = Counter()
    agent_interactions = defaultdict(set)

    for step in steps:
        if "messages" in step:
            for msg in step["messages"]:
                msg_type = msg.get("type", "unknown")
                message_types[msg_type] += 1

                sender = msg.get("sender_id")
                receiver = msg.get("receiver_id", -1)
                if sender is not None:
                    agent_interactions[sender].add(receiver)

    # Calculate conversation depth
    conversation_depth = 0
    for step in steps:
        if "messages" in step and len(step["messages"]) > 0:
            conversation_depth += 1

    return {
        "messages_per_step": messages_per_step,
        "message_types": dict(message_types),
        "agent_interactions": {k: list(v) for k, v in agent_interactions.items()},
        "conversation_depth": conversation_depth,
        "avg_messages_per_step": (
            sum(messages_per_step) / len(messages_per_step) if messages_per_step else 0
        ),
    }


def analyze_collusion_detection(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze collusion detection results."""
    collusive_messages = 0
    total_confidence = 0
    confidence_count = 0

    for step in episode_data["steps"]:
        if "chat_monitoring" in step:
            chat_monitoring = step["chat_monitoring"]
            collusive_messages += chat_monitoring.get("collusive_messages", 0)

            if "analysis_results" in chat_monitoring:
                for analysis in chat_monitoring["analysis_results"]:
                    confidence = analysis.get("confidence", 0)
                    total_confidence += confidence
                    confidence_count += 1

    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0

    return {
        "total_collusive_messages": collusive_messages,
        "average_confidence": avg_confidence,
        "total_analyzed": confidence_count,
    }


def view_messages(episode_data: Dict[str, Any], limit: Optional[int] = None) -> None:
    """Display messages from the episode."""
    all_messages = []

    for step in episode_data["steps"]:
        if "messages" in step:
            for msg in step["messages"]:
                all_messages.append(
                    {
                        "step": step["step"],
                        "sender": msg.get("sender_id", "unknown"),
                        "receiver": msg.get("receiver_id", -1),
                        "content": msg.get("content", ""),
                        "type": msg.get("type", "unknown"),
                    }
                )

    if limit:
        all_messages = all_messages[:limit]

    print(f"\n📝 Messages ({len(all_messages)} total):")
    print("-" * 80)

    for msg in all_messages:
        receiver_str = (
            f" -> {msg['receiver']}" if msg["receiver"] != -1 else " (broadcast)"
        )
        print(f"Step {msg['step']}: Agent {msg['sender']}{receiver_str}")
        print(f"  Type: {msg['type']}")
        print(f"  Content: {msg['content']}")
        print()


def print_analysis_summary(
    episode_data: Dict[str, Any], analysis_results: Dict[str, Any]
) -> None:
    """Print comprehensive analysis summary."""
    print("📊 Episode Analysis Summary")
    print("=" * 60)

    # Basic episode info
    if episode_data["header"]:
        header = episode_data["header"]
        print(f"Episode ID: {header.get('episode_id', 'unknown')}")
        print(f"Firms: {header.get('n_firms', 'unknown')}")
        print(f"Steps: {header.get('n_steps', 'unknown')}")
        print(f"Agent Types: {', '.join(header.get('agent_types', []))}")

    print(f"\nTotal Steps: {len(episode_data['steps'])}")

    # Message analysis
    msg_analysis = analysis_results["messages"]
    print("\n💬 Message Analysis:")
    print(f"  Total Messages: {msg_analysis['total_messages']}")
    print(f"  Message Types: {msg_analysis['message_types']}")

    # Conversation realism
    conv_analysis = analysis_results["conversation"]
    print("\n🎭 Conversation Realism:")
    print(f"  Conversation Depth: {conv_analysis['conversation_depth']} steps")
    print(f"  Avg Messages/Step: {conv_analysis['avg_messages_per_step']:.2f}")
    print(f"  Agent Interactions: {len(conv_analysis['agent_interactions'])} agents")

    # Collusion detection
    collusion_analysis = analysis_results["collusion"]
    print("\n🚨 Collusion Detection:")
    print(f"  Collusive Messages: {collusion_analysis['total_collusive_messages']}")
    print(f"  Average Confidence: {collusion_analysis['average_confidence']:.2f}")
    print(f"  Total Analyzed: {collusion_analysis['total_analyzed']}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze episode data")
    parser.add_argument("file", help="Episode JSONL file to analyze")
    parser.add_argument("--view-messages", action="store_true", help="Display messages")
    parser.add_argument("--limit", type=int, help="Limit number of messages to display")
    parser.add_argument("--summary-only", action="store_true", help="Show only summary")

    args = parser.parse_args()

    # Load episode data
    try:
        episode_data = load_episode_data(args.file)
    except Exception as e:
        print(f"Error loading episode data: {e}")
        return 1

    # Perform analysis
    analysis_results = {
        "messages": analyze_messages(episode_data),
        "conversation": analyze_conversation_realism(episode_data),
        "collusion": analyze_collusion_detection(episode_data),
    }

    # Display results
    if not args.summary_only:
        print_analysis_summary(episode_data, analysis_results)

    if args.view_messages:
        view_messages(episode_data, args.limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
