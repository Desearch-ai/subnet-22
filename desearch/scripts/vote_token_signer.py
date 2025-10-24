import jwt
import time
import json
from bittensor import wallet as btcli_wallet
from substrateinterface import Keypair

CYAN = "\033[96m" # field color
GREEN = "\033[92m" # indicating success
RED = "\033[91m" # indicating error
RESET = "\033[0m" # resetting color to the default
DIVIDER = '-' * 86


def validate_jwt(token):
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        if int(time.time()) > decoded["exp"]:
            return f"{RED}Access Token is expired. Please generate a new access token.{RESET}"
        return None
    except jwt.DecodeError:
        return f"{RED}Invalid JWT format.{RESET}"
    except KeyError:
        return f"""{RED}Expiration claim ('exp') not found in the JWT payload.
Please go to the {CYAN}vote.datura.ai{RESET} and generate new access key{RED}{RESET}"""
    except ValueError:
        return f"""{RED}Invalid expiration time format in the JWT payload.
Please go to the {CYAN}vote.datura.ai{RESET} and generate new access key{RED}{RESET}"""


def run():
    key = input(f"{CYAN}Enter the access key to be signed: {RESET}")
    access_key_error = validate_jwt(key)
    if access_key_error:
       print(access_key_error)
       exit(1)

    name = input(f"{CYAN}Enter wallet name (default: Coldkey): {RESET}") or "Coldkey"
    path = input(f"{CYAN}Enter wallet path (default: ~/.bittensor/wallets/): {RESET}") or "~/.bittensor/wallets/"

    wallet = btcli_wallet(name=name, path=path)
    try:
        coldkey = wallet.get_coldkey()
    except Exception as e:
        print(f"{RED}Error loading coldkey: {e} {RESET}")
        exit(1)

    signature = coldkey.sign(key.encode("utf-8")).hex()
    keypair = Keypair(ss58_address=coldkey.ss58_address)
    is_valid = keypair.verify(key.encode("utf-8"), bytes.fromhex(signature))
    if not is_valid:
       print(f"{RED}Signature is not valid{RESET}")
       exit(1)

    vote_data = {"access_key": key, "signature": signature, "coldkey_ss58": coldkey.ss58_address}
    print(f'{CYAN}{DIVIDER}{RESET}')
    message = f"""\nCompleted the signature signing process, here is your credentials:
You can go to the {CYAN}vote.datura.ai{RESET} and paste your credentials and authorize.\n
{CYAN}Please note this credentials is valid for one week, after that time, you gonna need to\nre-sign credentials.\n"""
    print(f"{message}{RESET}")
    print(json.dumps(vote_data, indent=4))
    print('\n')



if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        exit()

