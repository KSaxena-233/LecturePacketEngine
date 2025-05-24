from lecture_packet_engine import LecturePacketEngine

def main():
    # Initialize the engine
    engine = LecturePacketEngine()
    
    # Example topics
    topics = [
        "Introduction to Quantum Computing",
        "Machine Learning Fundamentals",
        "The Theory of Relativity"
    ]
    
    # Generate lecture packets for each topic
    for topic in topics:
        try:
            print(f"\nGenerating lecture packet for: {topic}")
            output_path = engine.create_lecture_packet(topic)
            print(f"Successfully generated: {output_path}")
        except Exception as e:
            print(f"Error generating lecture packet for {topic}: {e}")

if __name__ == "__main__":
    main() 