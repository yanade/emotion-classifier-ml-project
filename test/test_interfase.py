from src.simple_interface import clean_text

def test_clean_text_removes_symbols():
    text = "Hello!!! ###World"
    cleaned = clean_text(text)
    assert "###" not in cleaned
    assert cleaned.strip() != ""
    assert cleaned == "hello!!! world"