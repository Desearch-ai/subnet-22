import sys
import bittensor as bt


def generate_token(value: str, coldkey: str):
    try:
        wallet = bt.wallet(name=coldkey)
        signature = wallet.coldkey.sign(value.encode()).hex()
        print(f"\nSignature:\n  {signature}")
        print(f"\nColdkey Address:\n  {wallet.coldkey.ss58_address}\n")
    except Exception as e:
        print(f"[!] Error: {e}")


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 3:
        args = '[access_key] [coldkey]'
        print(f'\nUsage:\n python token_generator.py {args}\n')
        sys.exit(1)

    value = sys.argv[1]
    coldkey = sys.argv[2]
    generate_token(value, coldkey)
