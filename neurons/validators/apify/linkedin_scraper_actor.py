import os
from typing import List, Optional

# DEALINGS IN THE SOFTWARE.p
import traceback
import bittensor as bt
from apify_client import ApifyClientAsync
from datura.protocol import (
    PeopleSearchResult,
    LinkedinEducationItem,
    LinkedinExperienceItem,
    LinkedinLanguageItem,
)


APIFY_API_KEY = os.environ.get("APIFY_API_KEY")

# todo at ths moment just warning, later it will be required
if not APIFY_API_KEY:
    raise ValueError(
        "Please set the APIFY_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )


def toLinkedinExperienceItem(item):
    return LinkedinExperienceItem(
        **item,
        company_id=item.get("companyId"),
        company_link=item.get("companyLink1"),
    )


def toLinkedinEducationItem(item):
    return LinkedinEducationItem(
        **item,
        company_id=item.get("companyId"),
        company_link=item.get("companyLink1"),
    )


def toPeopleSearchResult(item):
    if item is None:
        return None

    profile = PeopleSearchResult(
        full_name=item.get("fullName"),
        avatar=item.get("profilePic"),
        link=item.get("linkedinUrl"),
        title=item.get("headline"),
        summary=item.get("about"),
        first_name=item.get("firstName"),
        last_name=item.get("lastName"),
        experiences=[
            toLinkedinExperienceItem(experience)
            for experience in (item.get("experiences") or [])
        ],
        educations=[
            toLinkedinEducationItem(experience)
            for experience in (item.get("educations") or [])
        ],
        languages=item.get("languages"),
    )

    return profile


class LinkedinScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/dev_fusion/linkedin-profile-scraper
        self.actor_id = "2SyF0bVxmgGr8IVCZ"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def get_profiles(self, urls: List[str]) -> List[PeopleSearchResult]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []
        try:
            run_input = {
                "profileUrls": urls,
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            profiles: List[PeopleSearchResult] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                try:
                    if item.get("noResults"):
                        continue

                    profiles.append(toPeopleSearchResult(item).model_dump())
                except Exception as e:
                    error_message = (
                        f"LinkedinScraperActor: Failed to scrape profile: {str(e)}"
                    )
                    tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                    bt.logging.warning("\n".join(tb_str) + error_message)

            return profiles
        except Exception as e:
            error_message = (
                f"LinkedinScraperActor: Failed to scrape profile {urls}: {str(e)}"
            )
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            return []
