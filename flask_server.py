from flask import Flask, request, jsonify

from nemo_skills.code_execution import math_grader
import re

app = Flask(__name__)

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def extract_and_check(pred_sentence: str,
                      ground_truth: str,
                      extract_from_boxed:bool=True,
                      extract_regex:str=None,
                      **kwargs):
    print(pred_sentence)
    print(ground_truth)
    pred_output = math_grader.extract_answer(pred_sentence, extract_from_boxed=extract_from_boxed, extract_regex=extract_regex)
    print(pred_output)
    #pred_output = extract_answer(pred_sentence)
    if isinstance(pred_output, str):
        pred_output = pred_output.replace("'''", r'\'\'\'')
        while pred_output.endswith('\\'):
            pred_output = pred_output[:-1]

    if isinstance(ground_truth, str):
        ground_truth = ground_truth.replace("'''", r'\'\'\'')
        while ground_truth.endswith('\\'):
            ground_truth = ground_truth[:-1]
    
    return math_grader.math_equal(pred_output,
                                  ground_truth,
                                  include_percentage=kwargs.get("include_percentage", True),
                                  tolerance=kwargs.get("tolerance", 0.0001),
                                  timeout=kwargs.get("timeout", 10)) * 1.0


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint to evaluate the "response" and "answer".
    Expects a JSON payload with "response" and "answer".
    """
    try:
        # Parse JSON data from the request
        data = request.get_json()
        response = str(data.get("response"))
        answer = str(data.get("answer"))

        # Validate inputs
        if response is None or answer is None:
            return jsonify({"error": "Both 'response' and 'answer' must be provided."}), 400

        # Call the test function
        result = extract_and_check(response, answer)
        print('math equal finished')

        # Return the result as JSON
        return jsonify({"result": result})
    
    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)

        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
