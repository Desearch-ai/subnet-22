# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# version must stay on line 22
__version__ = "0.0.195"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

u64_max = 2**64 - 9
__weights_version__ = u64_max

print("__version__", __version__)

import os
from openai import AsyncOpenAI
from enum import Enum

AsyncOpenAI.api_key = os.environ.get("OPENAI_API_KEY")
if not AsyncOpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=90.0)

# Blacklist variables
ALLOW_NON_REGISTERED = False
PROMPT_BLACKLIST_STAKE = 20000
TWITTER_SCRAPPER_BLACKLIST_STAKE = 20000
ISALIVE_BLACKLIST_STAKE = min(PROMPT_BLACKLIST_STAKE, TWITTER_SCRAPPER_BLACKLIST_STAKE)
MIN_REQUEST_PERIOD = 2
MAX_REQUESTS = 30
# must have the test_key whitelisted to avoid a global blacklist
testnet_key = ["5EhEZN6soubtKJm8RN7ANx9FGZ2JezxBUFxr45cdsHtDp3Uk"]
test_key = ["5DcRHcCwD33YsHfj4PX5j2evWLniR1wSWeNmpf5RXaspQT6t"]

valid_validators = [
    "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v",
    "5EhvL1FVkQPpMjZX4MAADcW42i3xPSF1KiCpuaxTYVr28sux",
    "5CXRfP2ekFhe62r7q3vppRajJmGhTi7vwvb2yr79jveZ282w",
    "5CaNj3BarTHotEK1n513aoTtFeXcjf6uvKzAyzNuv9cirUoW",
    "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN",
    "5DvTpiniW9s3APmHRYn8FroUWyfnLtrsid5Mtn5EwMXHN2ed",
    "5G3f8VDTT1ydirT3QffnV2TMrNMR2MkQfGUubQNqZcGSj82T",
    "5Dz8ShM6rtPw1GBAaqxjycT9LF1TC3iDpzpUH9gKr85Nizo6",
    "5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8",
    "5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf",
    "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2",
    "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3",
    "5H66kJAzBCv2DC9poHATLQqyt3ag8FLSbHf6rMqTiRcS52rc",
    "5HbLYXUBy1snPR8nfioQ7GoA9x76EELzEq9j7F32vWUQHm1x",
    "5FKstHjZkh4v3qAMSBa1oJcHCLjxYZ8SNTSz1opTv4hR7gVB",
    "5DXTJSPVvf1sow1MU4npJPewEAwhPRb6CWsk4RX9RFt2PRbj",
    "5EsrMfo7UcPs6AqAotU47VmYGfLHntS9JzhEwbY2EJMcWQxQ",  # server
    "5Dd8gaRNdhm1YP7G1hcB1N842ecAUQmbLjCRLqH5ycaTGrWv",
    "5DnXm2tBGAD57ySJv5SfpTfLcsQbSKKp6xZKFWABw3cYUgqg",
    "5GVpVH7DjYmQY7ckznVnrHncU9knzYJvhY3TfbFY7sPboJB2",
    "5Fq5v71D4LX8Db1xsmRSy6udQThcZ8sFDqxQFwnUZ1BuqY5A",
    "5ChuGqW2cxc5AZJ29z6vyTkTncg75L9ovfp8QN8eB8niSD75",
    "5F27Eqz2PhyMtGMEce898x31DokNqRVxkm5AhDDe6rDGNvoY",
]

WHITELISTED_KEYS = testnet_key + test_key + valid_validators

BLACKLISTED_KEYS = [
    "5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe",
    "5GsL9zNp1CdKmKSYBGjeF9kGReRpS8KdQv2yJ3mHVwKq2YCq",
]

ENTITY = "smart-scrape"
PROJECT_NAME = "smart-scrape-1.0"


class QUERY_MINERS(Enum):
    ALL = "all"
    RANDOM = "random"


# Import all submodules.
from . import protocol
from . import utils
from . import tools
