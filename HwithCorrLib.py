from transformers import MarianMTModel, MarianTokenizer
import re
import language_tool_python

# LanguageTool for Arabic grammar correction
try:
    arabic_tool = language_tool_python.LanguageTool("ar")
except Exception as e:
    print(f"Error initializing LanguageTool: {e}")
    exit()

def translate_helsinki(text, src_lang="en", tgt_lang="ar"):
    model_name = "Helsinki-NLP/opus-mt-en-ar"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    generated_tokens = model.generate(**encoded_text)

    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def load_correction_rules(filepath):
    rules = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):  
                    parts = line.split("=")
                    if len(parts) == 3:
                        _, original, replacement = parts
                        rules.append((original.strip(), replacement.strip()))
    except FileNotFoundError:
        print(f"Warning: Rules file '{filepath}' not found.")
    return rules

def english_to_arabic_numerals(text):
    """Converts English numerals (0-9) to Arabic numerals (٠-٩)."""
    translation_table = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    return text.translate(translation_table)

def correct_translation(translated_text, rules):
    """Applies regex-based correction rules and fixes formatting issues."""
    
    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\+\d{11}",
        "credit_card": r"\d{4}-\d{4}-\d{4}-\d{4}",
        "ssn": r"\d{3}-\d{2}-\d{4}"
    }

    # Extract PII to restore later
    pii_data = {key: re.findall(pattern, translated_text) for key, pattern in patterns.items()}

    # Apply regex correction rules
    for original, replacement in rules:
        translated_text = re.sub(original, replacement, translated_text)

    # Convert numerals
    translated_text = english_to_arabic_numerals(translated_text)

    # Correct Arabic grammar using LanguageTool
    matches = arabic_tool.check(translated_text)
    translated_text = language_tool_python.utils.correct(translated_text, matches)

    # Restore PII after fixing numerals
    for key, pattern in patterns.items():
        for pii_value in pii_data[key]:
            translated_text = re.sub(pattern, pii_value, translated_text, 1)

    return translated_text

correction_file_path = r"D:\BigTapp\Company Work\.venv\English to Arab\Datasets\CorrectionCode.txt"

rules = load_correction_rules(correction_file_path)

# Example usage
if __name__ == "__main__":
    input_text = """Michael Johnson is a software engineer at Google. His email is michael.johnson@example.com, and his phone number is +14155552678. His website is www.michaeljohnson.dev. His credit card number is 1234-5678-9012-3456, and his social security number is 321-54-9876."""

    translated_text = translate_helsinki(input_text)
    corrected_text = correct_translation(translated_text, rules)

    print("Original:", input_text)
    print("Translated:", translated_text)
    print("Corrected:", corrected_text)
