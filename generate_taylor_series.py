from lecture_packet_engine import LecturePacketEngine
import os

def main():
    engine = LecturePacketEngine()
    prompt = "Photosynthesis: Process, Pathways, and Applications"
    try:
        output_path = engine.create_lecture_packet(prompt)
        # Rename the output file to koala.pdf
        os.rename(output_path, "koala.pdf")
        print("Photosynthesis lecture packet generated successfully: koala.pdf")
    except Exception as e:
        print(f"Error generating lecture packet: {e}")

if __name__ == "__main__":
    main() 