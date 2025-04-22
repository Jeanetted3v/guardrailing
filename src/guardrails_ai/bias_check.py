"""To run:
python -m src.guardrails_ai.bias_check
"""
from guardrails.hub import BiasCheck
from guardrails import Guard


def test_bias_check_simple():
    """
    A minimal test for BiasCheck validator following the documentation exactly.
    """
    guard = Guard().use(
        BiasCheck(threshold=0.9, on_fail="exception")
    )

    try:
        response = guard.validate("The movie was great.")
    except Exception as e:
        print(e)

    # print("✅ PASSED as expected")

    # # Test the failing example
    # print("\nTesting: 'Why do men think that movie was great?'")
    # try:
    #     result = guard.validate("Why do men think that movie was great?")
    #     print("❌ PASSED unexpectedly - should have failed")
    # except Exception as e:
    #     print("✅ FAILED as expected")
    #     print(f"Error message: {str(e)}")
    
    # print("\nTest complete!")

if __name__ == "__main__":
    test_bias_check_simple()