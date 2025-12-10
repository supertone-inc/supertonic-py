"""
Example 7: Text Validation

Check if text can be processed and identify unsupported characters.

TODO: This example uses methods that don't exist in UnicodeProcessor yet.
      Need to implement the following methods:
      - validate_text(text, preprocess=False) - add preprocess parameter
      - get_unsupported_characters(text)
      - get_supported_characters()
      - get_supported_unicode_ranges()
"""

# import os
#
# from supertonic import TTS
#
# os.makedirs("outputs/test7", exist_ok=True)
#
# tts = TTS()
#
# # Get the text processor
# text_processor = tts.model.text_processor
#
# # Example texts with various characters
# test_texts = [
#     "Hello World! This is a normal English sentence.",
#     "Text with emojis 😀🎉 and special chars",
#     "Mixed text: Hello 世界",
#     "Special punctuation: e.g., test @ test & more",
# ]
#
# print("=" * 70)
# print("Text Validation Examples")
# print("=" * 70)
#
# for i, text in enumerate(test_texts, 1):
#     print(f"\n{i}. Testing: {text[:50]}...")
#
#     # Validate before preprocessing
#     is_valid, unsupported = text_processor.validate_text(text, preprocess=False)
#
#     if not is_valid:
#         print(f"   ⚠️  Contains {len(unsupported)} unsupported character(s): {unsupported[:5]}")
#     else:
#         print("   ✓ All characters supported")
#
#     # Show preprocessed version
#     preprocessed = text_processor._preprocess_text(text)
#     if preprocessed != text:
#         print(f"   After preprocessing: {preprocessed[:50]}...")
#
#     # Get unsupported characters only
#     unsupported_chars = text_processor.get_unsupported_characters(text)
#     if unsupported_chars:
#         print(f"   Unsupported chars: {sorted(list(unsupported_chars))[:10]}")
#
# # Show supported character stats
# supported = text_processor.get_supported_characters()
# print(f"\n{'=' * 70}")
# print(f"Model supports {len(supported)} unique characters")
#
# # Show unicode ranges
# ranges = text_processor.get_supported_unicode_ranges()
# print(f"Supported unicode ranges: {len(ranges)} range(s)")
# for start, end in ranges[:3]:  # Show first 3 ranges
#     print(f"  U+{start:04X} to U+{end:04X}")
# if len(ranges) > 3:
#     print(f"  ... and {len(ranges) - 3} more range(s)")
