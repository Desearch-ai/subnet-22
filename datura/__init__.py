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
__version__ = "0.0.177"
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


from enum import Enum


class DateFilter(Enum):
    ANY_TIME = "any time"
    PAST_HOUR = "past hour"
    PAST_24_HOURS = "past 24 hours"
    PAST_2_DAYS = "past 2 days"
    PAST_WEEK = "past week"
    PAST_MONTH = "past month"
    PAST_YEAR = "past year"
    CUSTOM = "custom"


region = [
    {"country_code": "af", "country_name": "Afghanistan"},
    {"country_code": "al", "country_name": "Albania"},
    {"country_code": "dz", "country_name": "Algeria"},
    {"country_code": "as", "country_name": "American Samoa"},
    {"country_code": "ad", "country_name": "Andorra"},
    {"country_code": "ao", "country_name": "Angola"},
    {"country_code": "ai", "country_name": "Anguilla"},
    {"country_code": "aq", "country_name": "Antarctica"},
    {"country_code": "ag", "country_name": "Antigua and Barbuda"},
    {"country_code": "ar", "country_name": "Argentina"},
    {"country_code": "am", "country_name": "Armenia"},
    {"country_code": "aw", "country_name": "Aruba"},
    {"country_code": "au", "country_name": "Australia"},
    {"country_code": "at", "country_name": "Austria"},
    {"country_code": "az", "country_name": "Azerbaijan"},
    {"country_code": "bs", "country_name": "Bahamas"},
    {"country_code": "bh", "country_name": "Bahrain"},
    {"country_code": "bd", "country_name": "Bangladesh"},
    {"country_code": "bb", "country_name": "Barbados"},
    {"country_code": "by", "country_name": "Belarus"},
    {"country_code": "be", "country_name": "Belgium"},
    {"country_code": "bz", "country_name": "Belize"},
    {"country_code": "bj", "country_name": "Benin"},
    {"country_code": "bm", "country_name": "Bermuda"},
    {"country_code": "bt", "country_name": "Bhutan"},
    {"country_code": "bo", "country_name": "Bolivia"},
    {"country_code": "ba", "country_name": "Bosnia and Herzegovina"},
    {"country_code": "bw", "country_name": "Botswana"},
    {"country_code": "bv", "country_name": "Bouvet Island"},
    {"country_code": "br", "country_name": "Brazil"},
    {"country_code": "io", "country_name": "British Indian Ocean Territory"},
    {"country_code": "bn", "country_name": "Brunei Darussalam"},
    {"country_code": "bg", "country_name": "Bulgaria"},
    {"country_code": "bf", "country_name": "Burkina Faso"},
    {"country_code": "bi", "country_name": "Burundi"},
    {"country_code": "kh", "country_name": "Cambodia"},
    {"country_code": "cm", "country_name": "Cameroon"},
    {"country_code": "ca", "country_name": "Canada"},
    {"country_code": "cv", "country_name": "Cape Verde"},
    {"country_code": "ky", "country_name": "Cayman Islands"},
    {"country_code": "cf", "country_name": "Central African Republic"},
    {"country_code": "td", "country_name": "Chad"},
    {"country_code": "cl", "country_name": "Chile"},
    {"country_code": "cn", "country_name": "China"},
    {"country_code": "cx", "country_name": "Christmas Island"},
    {"country_code": "cc", "country_name": "Cocos (Keeling) Islands"},
    {"country_code": "co", "country_name": "Colombia"},
    {"country_code": "km", "country_name": "Comoros"},
    {"country_code": "cg", "country_name": "Congo"},
    {"country_code": "cd", "country_name": "Congo, the Democratic Republic of the"},
    {"country_code": "ck", "country_name": "Cook Islands"},
    {"country_code": "cr", "country_name": "Costa Rica"},
    {"country_code": "ci", "country_name": "Cote D'ivoire"},
    {"country_code": "hr", "country_name": "Croatia"},
    {"country_code": "cu", "country_name": "Cuba"},
    {"country_code": "cy", "country_name": "Cyprus"},
    {"country_code": "cz", "country_name": "Czech Republic"},
    {"country_code": "dk", "country_name": "Denmark"},
    {"country_code": "dj", "country_name": "Djibouti"},
    {"country_code": "dm", "country_name": "Dominica"},
    {"country_code": "do", "country_name": "Dominican Republic"},
    {"country_code": "ec", "country_name": "Ecuador"},
    {"country_code": "eg", "country_name": "Egypt"},
    {"country_code": "sv", "country_name": "El Salvador"},
    {"country_code": "gq", "country_name": "Equatorial Guinea"},
    {"country_code": "er", "country_name": "Eritrea"},
    {"country_code": "ee", "country_name": "Estonia"},
    {"country_code": "et", "country_name": "Ethiopia"},
    {"country_code": "fk", "country_name": "Falkland Islands (Malvinas)"},
    {"country_code": "fo", "country_name": "Faroe Islands"},
    {"country_code": "fj", "country_name": "Fiji"},
    {"country_code": "fi", "country_name": "Finland"},
    {"country_code": "fr", "country_name": "France"},
    {"country_code": "gf", "country_name": "French Guiana"},
    {"country_code": "pf", "country_name": "French Polynesia"},
    {"country_code": "tf", "country_name": "French Southern Territories"},
    {"country_code": "ga", "country_name": "Gabon"},
    {"country_code": "gm", "country_name": "Gambia"},
    {"country_code": "ge", "country_name": "Georgia"},
    {"country_code": "de", "country_name": "Germany"},
    {"country_code": "gh", "country_name": "Ghana"},
    {"country_code": "gi", "country_name": "Gibraltar"},
    {"country_code": "gr", "country_name": "Greece"},
    {"country_code": "gl", "country_name": "Greenland"},
    {"country_code": "gd", "country_name": "Grenada"},
    {"country_code": "gp", "country_name": "Guadeloupe"},
    {"country_code": "gu", "country_name": "Guam"},
    {"country_code": "gt", "country_name": "Guatemala"},
    {"country_code": "gn", "country_name": "Guinea"},
    {"country_code": "gw", "country_name": "Guinea-Bissau"},
    {"country_code": "gy", "country_name": "Guyana"},
    {"country_code": "ht", "country_name": "Haiti"},
    {"country_code": "hm", "country_name": "Heard Island and Mcdonald Islands"},
    {"country_code": "va", "country_name": "Holy See (Vatican City State)"},
    {"country_code": "hn", "country_name": "Honduras"},
    {"country_code": "hk", "country_name": "Hong Kong"},
    {"country_code": "hu", "country_name": "Hungary"},
    {"country_code": "is", "country_name": "Iceland"},
    {"country_code": "in", "country_name": "India"},
    {"country_code": "id", "country_name": "Indonesia"},
    {"country_code": "ir", "country_name": "Iran, Islamic Republic of"},
    {"country_code": "iq", "country_name": "Iraq"},
    {"country_code": "ie", "country_name": "Ireland"},
    {"country_code": "il", "country_name": "Israel"},
    {"country_code": "it", "country_name": "Italy"},
    {"country_code": "jm", "country_name": "Jamaica"},
    {"country_code": "jp", "country_name": "Japan"},
    {"country_code": "jo", "country_name": "Jordan"},
    {"country_code": "kz", "country_name": "Kazakhstan"},
    {"country_code": "ke", "country_name": "Kenya"},
    {"country_code": "ki", "country_name": "Kiribati"},
    {"country_code": "kp", "country_name": "Korea, Democratic People's Republic of"},
    {"country_code": "kr", "country_name": "Korea, Republic of"},
    {"country_code": "kw", "country_name": "Kuwait"},
    {"country_code": "kg", "country_name": "Kyrgyzstan"},
    {"country_code": "la", "country_name": "Lao People's Democratic Republic"},
    {"country_code": "lv", "country_name": "Latvia"},
    {"country_code": "lb", "country_name": "Lebanon"},
    {"country_code": "ls", "country_name": "Lesotho"},
    {"country_code": "lr", "country_name": "Liberia"},
    {"country_code": "ly", "country_name": "Libyan Arab Jamahiriya"},
    {"country_code": "li", "country_name": "Liechtenstein"},
    {"country_code": "lt", "country_name": "Lithuania"},
    {"country_code": "lu", "country_name": "Luxembourg"},
    {"country_code": "mo", "country_name": "Macao"},
    {
        "country_code": "mk",
        "country_name": "Macedonia, the Former Yugosalv Republic of",
    },
    {"country_code": "mg", "country_name": "Madagascar"},
    {"country_code": "mw", "country_name": "Malawi"},
    {"country_code": "my", "country_name": "Malaysia"},
    {"country_code": "mv", "country_name": "Maldives"},
    {"country_code": "ml", "country_name": "Mali"},
    {"country_code": "mt", "country_name": "Malta"},
    {"country_code": "mh", "country_name": "Marshall Islands"},
    {"country_code": "mq", "country_name": "Martinique"},
    {"country_code": "mr", "country_name": "Mauritania"},
    {"country_code": "mu", "country_name": "Mauritius"},
    {"country_code": "yt", "country_name": "Mayotte"},
    {"country_code": "mx", "country_name": "Mexico"},
    {"country_code": "fm", "country_name": "Micronesia, Federated States of"},
    {"country_code": "md", "country_name": "Moldova, Republic of"},
    {"country_code": "mc", "country_name": "Monaco"},
    {"country_code": "mn", "country_name": "Mongolia"},
    {"country_code": "ms", "country_name": "Montserrat"},
    {"country_code": "ma", "country_name": "Morocco"},
    {"country_code": "mz", "country_name": "Mozambique"},
    {"country_code": "mm", "country_name": "Myanmar"},
    {"country_code": "na", "country_name": "Namibia"},
    {"country_code": "nr", "country_name": "Nauru"},
    {"country_code": "np", "country_name": "Nepal"},
    {"country_code": "nl", "country_name": "Netherlands"},
    {"country_code": "an", "country_name": "Netherlands Antilles"},
    {"country_code": "nc", "country_name": "New Caledonia"},
    {"country_code": "nz", "country_name": "New Zealand"},
    {"country_code": "ni", "country_name": "Nicaragua"},
    {"country_code": "ne", "country_name": "Niger"},
    {"country_code": "ng", "country_name": "Nigeria"},
    {"country_code": "nu", "country_name": "Niue"},
    {"country_code": "nf", "country_name": "Norfolk Island"},
    {"country_code": "mp", "country_name": "Northern Mariana Islands"},
    {"country_code": "no", "country_name": "Norway"},
    {"country_code": "om", "country_name": "Oman"},
    {"country_code": "pk", "country_name": "Pakistan"},
    {"country_code": "pw", "country_name": "Palau"},
    {"country_code": "ps", "country_name": "Palestinian Territory, Occupied"},
    {"country_code": "pa", "country_name": "Panama"},
    {"country_code": "pg", "country_name": "Papua New Guinea"},
    {"country_code": "py", "country_name": "Paraguay"},
    {"country_code": "pe", "country_name": "Peru"},
    {"country_code": "ph", "country_name": "Philippines"},
    {"country_code": "pn", "country_name": "Pitcairn"},
    {"country_code": "pl", "country_name": "Poland"},
    {"country_code": "pt", "country_name": "Portugal"},
    {"country_code": "pr", "country_name": "Puerto Rico"},
    {"country_code": "qa", "country_name": "Qatar"},
    {"country_code": "re", "country_name": "Reunion"},
    {"country_code": "ro", "country_name": "Romania"},
    {"country_code": "ru", "country_name": "Russian Federation"},
    {"country_code": "rw", "country_name": "Rwanda"},
    {"country_code": "sh", "country_name": "Saint Helena"},
    {"country_code": "kn", "country_name": "Saint Kitts and Nevis"},
    {"country_code": "lc", "country_name": "Saint Lucia"},
    {"country_code": "pm", "country_name": "Saint Pierre and Miquelon"},
    {"country_code": "vc", "country_name": "Saint Vincent and the Grenadines"},
    {"country_code": "ws", "country_name": "Samoa"},
    {"country_code": "sm", "country_name": "San Marino"},
    {"country_code": "st", "country_name": "Sao Tome and Principe"},
    {"country_code": "sa", "country_name": "Saudi Arabia"},
    {"country_code": "sn", "country_name": "Senegal"},
    {"country_code": "rs", "country_name": "Serbia and Montenegro"},
    {"country_code": "sc", "country_name": "Seychelles"},
    {"country_code": "sl", "country_name": "Sierra Leone"},
    {"country_code": "sg", "country_name": "Singapore"},
    {"country_code": "sk", "country_name": "Slovakia"},
    {"country_code": "si", "country_name": "Slovenia"},
    {"country_code": "sb", "country_name": "Solomon Islands"},
    {"country_code": "so", "country_name": "Somalia"},
    {"country_code": "za", "country_name": "South Africa"},
    {
        "country_code": "gs",
        "country_name": "South Georgia and the South Sandwich Islands",
    },
    {"country_code": "es", "country_name": "Spain"},
    {"country_code": "lk", "country_name": "Sri Lanka"},
    {"country_code": "sd", "country_name": "Sudan"},
    {"country_code": "sr", "country_name": "Suriname"},
    {"country_code": "sj", "country_name": "Svalbard and Jan Mayen"},
    {"country_code": "sz", "country_name": "Swaziland"},
    {"country_code": "se", "country_name": "Sweden"},
    {"country_code": "ch", "country_name": "Switzerland"},
    {"country_code": "sy", "country_name": "Syrian Arab Republic"},
    {"country_code": "tw", "country_name": "Taiwan, Province of China"},
    {"country_code": "tj", "country_name": "Tajikistan"},
    {"country_code": "tz", "country_name": "Tanzania, United Republic of"},
    {"country_code": "th", "country_name": "Thailand"},
    {"country_code": "tl", "country_name": "Timor-Leste"},
    {"country_code": "tg", "country_name": "Togo"},
    {"country_code": "tk", "country_name": "Tokelau"},
    {"country_code": "to", "country_name": "Tonga"},
    {"country_code": "tt", "country_name": "Trinidad and Tobago"},
    {"country_code": "tn", "country_name": "Tunisia"},
    {"country_code": "tr", "country_name": "Turkey"},
    {"country_code": "tm", "country_name": "Turkmenistan"},
    {"country_code": "tc", "country_name": "Turks and Caicos Islands"},
    {"country_code": "tv", "country_name": "Tuvalu"},
    {"country_code": "ug", "country_name": "Uganda"},
    {"country_code": "ua", "country_name": "Ukraine"},
    {"country_code": "ae", "country_name": "United Arab Emirates"},
    {"country_code": "uk", "country_name": "United Kingdom"},
    {"country_code": "gb", "country_name": "United Kingdom"},
    {"country_code": "us", "country_name": "United States"},
    {"country_code": "um", "country_name": "United States Minor Outlying Islands"},
    {"country_code": "uy", "country_name": "Uruguay"},
    {"country_code": "uz", "country_name": "Uzbekistan"},
    {"country_code": "vu", "country_name": "Vanuatu"},
    {"country_code": "ve", "country_name": "Venezuela"},
    {"country_code": "vn", "country_name": "Viet Nam"},
    {"country_code": "vg", "country_name": "Virgin Islands, British"},
    {"country_code": "vi", "country_name": "Virgin Islands, U.S."},
    {"country_code": "wf", "country_name": "Wallis and Futuna"},
    {"country_code": "eh", "country_name": "Western Sahara"},
    {"country_code": "ye", "country_name": "Yemen"},
    {"country_code": "zm", "country_name": "Zambia"},
    {"country_code": "zw", "country_name": "Zimbabwe"},
]
