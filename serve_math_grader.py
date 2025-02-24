from functools import partial

from nemo_skills.math_grader_server import SimpleMathGrader, extract_and_check

def main() -> None:
    print("STARTING GRADER SERVER1")
    server = SimpleMathGrader(
        grading_function=extract_and_check,
        port=5567,
        process_count=32,
    )
    print("GRADER SERVER CONSTRUCT")
    server.run_server()


if __name__ == "__main__":
    main()
